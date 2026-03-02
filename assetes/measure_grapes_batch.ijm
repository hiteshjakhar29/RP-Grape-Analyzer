// measure_grapes_batch.ijm — Batch grape color measurement (one ImageJ call).
//
// Called by Python:
//   fiji --headless --console -macro measure_grapes_batch.ijm "img|masks_dir|csv|N"
//
// Arguments (pipe-separated):
//   img       — absolute path to session image (PNG)
//   masks_dir — directory containing  1.png … N.png  (binary masks, white=grape)
//   csv       — output CSV path
//   N         — number of grapes (typically 36)
//
// Output CSV columns (one data row per grape):
//   grape_id, area_px,
//   mean_R, mean_G, mean_B,
//   mean_L, mean_a, mean_b,        (L* 0-100, a*/b* -128…+127)
//   mean_H, mean_S, mean_Br        (H 0-360°, S/Br 0-1)
//
// Strategy: pre-convert the session image ONCE into R/G/B channels, a Lab
// image, and an HSB stack — then for each grape simply apply the mask as a
// selection to each pre-computed image and call Measure.  This avoids
// per-grape duplicates / conversions and is much faster in batch mode.

args  = getArgument();
parts = split(args, "|");
img_path  = parts[0];
masks_dir = parts[1];
csv_path  = parts[2];
n_grapes  = parseInt(parts[3]);

setBatchMode(true);

// ── Open session image ────────────────────────────────────────────────────────
open(img_path);
orig_title = getTitle();
orig_id    = getImageID();

// ── Pre-split RGB channels ────────────────────────────────────────────────────
// Duplicate and split into three 8-bit grayscale windows.
selectImage(orig_id);
run("Duplicate...", "title=_rgb_src");
run("Split Channels");
// Resulting windows: "_rgb_src (red)", "_rgb_src (green)", "_rgb_src (blue)"
selectWindow("_rgb_src (red)");   r_id = getImageID();
selectWindow("_rgb_src (green)"); g_id = getImageID();
selectWindow("_rgb_src (blue)");  b_id = getImageID();

// ── Pre-convert to Lab ────────────────────────────────────────────────────────
// The Color Space Converter plugin may output 32-bit float (actual L*a*b* values)
// or 8-bit (scaled).  We check bit-depth per channel and descale if needed.
selectImage(orig_id);
run("Duplicate...", "title=_lab_src");
run("Color Space Converter", "from=RGB to=Lab");
lab_conv_id = getImageID();   // might be the same or a new image; grab current ID
lab_bd = bitDepth();          // check now, before split
run("Split Channels");
// Resulting windows: "_lab_src (C1)", "_lab_src (C2)", "_lab_src (C3)"
selectWindow("_lab_src (C1)"); lab_L_id = getImageID();
selectWindow("_lab_src (C2)"); lab_a_id = getImageID();
selectWindow("_lab_src (C3)"); lab_b_id = getImageID();

// ── Pre-convert to HSB stack ──────────────────────────────────────────────────
// HSB Stack always creates an 8-bit 3-slice stack:
//   slice 1 = Hue     (0-255 → multiply by 360/255 to get degrees)
//   slice 2 = Sat     (0-255 → divide by 255 to get 0-1)
//   slice 3 = Bri     (0-255 → divide by 255 to get 0-1)
selectImage(orig_id);
run("Duplicate...", "title=_hsb_src");
hsb_id = getImageID();
run("HSB Stack");

// ── Measurement settings (set once) ──────────────────────────────────────────
run("Set Measurements...", "area mean redirect=None decimal=6");

// ── Open output CSV ───────────────────────────────────────────────────────────
f = File.open(csv_path);
print(f, "grape_id,area_px,mean_R,mean_G,mean_B,mean_L,mean_a,mean_b,mean_H,mean_S,mean_Br");

// ── Main loop ─────────────────────────────────────────────────────────────────
for (g = 1; g <= n_grapes; g++) {

    mask_path = masks_dir + File.separator + g + ".png";
    if (!File.exists(mask_path)) {
        print(f, "" + g + ",0,0,0,0,0,0,0,0,0,0");
        continue;
    }

    // -- Load mask and create ROI --
    open(mask_path);
    mask_id = getImageID();
    setThreshold(128, 255);
    run("Create Selection");
    // At this point the selection is stored in global ROI memory.
    selectImage(mask_id);
    close();

    // -- Apply selection to each pre-computed image --
    selectImage(orig_id);   run("Restore Selection");
    selectImage(r_id);      run("Restore Selection");
    selectImage(g_id);      run("Restore Selection");
    selectImage(b_id);      run("Restore Selection");
    selectImage(lab_L_id);  run("Restore Selection");
    selectImage(lab_a_id);  run("Restore Selection");
    selectImage(lab_b_id);  run("Restore Selection");
    selectImage(hsb_id);    run("Restore Selection");

    // -- Area (pixel count on original) --
    selectImage(orig_id);
    run("Measure");
    area_val = getResult("Area", 0);
    run("Clear Results");

    // -- RGB means --
    selectImage(r_id);
    run("Measure");
    mr = getResult("Mean", 0);
    run("Clear Results");

    selectImage(g_id);
    run("Measure");
    mg = getResult("Mean", 0);
    run("Clear Results");

    selectImage(b_id);
    run("Measure");
    mb = getResult("Mean", 0);
    run("Clear Results");

    // -- Lab means (descale based on bit depth) --
    selectImage(lab_L_id);
    run("Measure");
    ml_raw = getResult("Mean", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        ml = ml_raw;           // 32-bit float: actual L* value
    } else {
        ml = ml_raw * 100.0 / 255.0;   // 8-bit scaled: L* = raw * 100/255
    }

    selectImage(lab_a_id);
    run("Measure");
    ma_raw = getResult("Mean", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        ma = ma_raw;
    } else {
        ma = ma_raw - 128.0;
    }

    selectImage(lab_b_id);
    run("Measure");
    mb_raw = getResult("Mean", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        mb_lab = mb_raw;
    } else {
        mb_lab = mb_raw - 128.0;
    }

    // -- HSB means (always 8-bit, always needs descaling) --
    selectImage(hsb_id);
    Stack.setSlice(1);
    run("Restore Selection");
    run("Measure");
    mh_raw = getResult("Mean", 0);
    mh = mh_raw * 360.0 / 255.0;
    run("Clear Results");

    Stack.setSlice(2);
    run("Restore Selection");
    run("Measure");
    ms_raw = getResult("Mean", 0);
    ms = ms_raw / 255.0;
    run("Clear Results");

    Stack.setSlice(3);
    run("Restore Selection");
    run("Measure");
    mbr_raw = getResult("Mean", 0);
    mbr = mbr_raw / 255.0;
    run("Clear Results");

    // -- Write row --
    print(f, "" + g + "," + area_val + "," +
             mr + "," + mg + "," + mb + "," +
             ml + "," + ma + "," + mb_lab + "," +
             mh + "," + ms + "," + mbr);
}

// ── Cleanup ───────────────────────────────────────────────────────────────────
File.close(f);
run("Close All");
setBatchMode(false);
eval("script", "System.exit(0);");
