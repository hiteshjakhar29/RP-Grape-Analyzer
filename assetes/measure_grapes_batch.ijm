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
//   mean_R, std_R, mean_G, std_G, mean_B, std_B,
//   mean_L, std_L, mean_a, std_a, mean_b, std_b,   (L* 0-100, a*/b* -128…+127)
//   mean_H, std_H, mean_S, std_S, mean_Br, std_Br  (H 0-360°, S/Br 0-1)
//
// All Mean and StdDev values are measured directly by ImageJ.
// StdDev scaling notes:
//   - RGB: raw 0-255, no scaling needed
//   - Lab 32-bit: actual L*a*b* units, no scaling
//   - Lab 8-bit: L std scaled by 100/255; a/b std unscaled (offset -128 does not affect spread)
//   - HSB: H std scaled by 360/255; S,Br std scaled by 1/255

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
selectImage(orig_id);
run("Duplicate...", "title=_rgb_src");
run("Split Channels");
selectWindow("_rgb_src (red)");   r_id = getImageID();
selectWindow("_rgb_src (green)"); g_id = getImageID();
selectWindow("_rgb_src (blue)");  b_id = getImageID();

// ── Pre-convert to Lab ────────────────────────────────────────────────────────
selectImage(orig_id);
run("Duplicate...", "title=_lab_src");
run("Color Space Converter", "from=RGB to=Lab");
lab_conv_id = getImageID();
lab_bd = bitDepth();
run("Split Channels");
selectWindow("_lab_src (C1)"); lab_L_id = getImageID();
selectWindow("_lab_src (C2)"); lab_a_id = getImageID();
selectWindow("_lab_src (C3)"); lab_b_id = getImageID();

// ── Pre-convert to HSB stack ──────────────────────────────────────────────────
selectImage(orig_id);
run("Duplicate...", "title=_hsb_src");
hsb_id = getImageID();
run("HSB Stack");

// ── Measurement settings — area + mean + standard deviation ──────────────────
run("Set Measurements...", "area mean standard redirect=None decimal=6");

// ── Open output CSV ───────────────────────────────────────────────────────────
f = File.open(csv_path);
print(f, "grape_id,area_px," +
         "mean_R,std_R,mean_G,std_G,mean_B,std_B," +
         "mean_L,std_L,mean_a,std_a,mean_b,std_b," +
         "mean_H,std_H,mean_S,std_S,mean_Br,std_Br");

// ── Main loop ─────────────────────────────────────────────────────────────────
for (g = 1; g <= n_grapes; g++) {

    mask_path = masks_dir + File.separator + g + ".png";
    if (!File.exists(mask_path)) {
        print(f, "" + g + ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0");
        continue;
    }

    // -- Load mask and create ROI --
    open(mask_path);
    mask_id = getImageID();
    setThreshold(128, 255);
    run("Create Selection");
    selectImage(mask_id);
    close();

    // -- Apply selection to all pre-computed images --
    selectImage(orig_id);   run("Restore Selection");
    selectImage(r_id);      run("Restore Selection");
    selectImage(g_id);      run("Restore Selection");
    selectImage(b_id);      run("Restore Selection");
    selectImage(lab_L_id);  run("Restore Selection");
    selectImage(lab_a_id);  run("Restore Selection");
    selectImage(lab_b_id);  run("Restore Selection");
    selectImage(hsb_id);    run("Restore Selection");

    // -- Area --
    selectImage(orig_id);
    run("Measure");
    area_val = getResult("Area", 0);
    run("Clear Results");

    // -- RGB mean + std (raw 0-255, no scaling needed) --
    selectImage(r_id);
    run("Measure");
    mr  = getResult("Mean",   0);
    sr  = getResult("StdDev", 0);
    run("Clear Results");

    selectImage(g_id);
    run("Measure");
    mg  = getResult("Mean",   0);
    sg  = getResult("StdDev", 0);
    run("Clear Results");

    selectImage(b_id);
    run("Measure");
    mb  = getResult("Mean",   0);
    sb  = getResult("StdDev", 0);
    run("Clear Results");

    // -- Lab mean + std --
    // Offset (-128) only affects Mean, not StdDev (constant shift leaves spread unchanged).
    // Scale (100/255 for L in 8-bit) affects both Mean and StdDev equally.
    selectImage(lab_L_id);
    run("Measure");
    ml_raw  = getResult("Mean",   0);
    sl_raw  = getResult("StdDev", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        ml = ml_raw;
        sl = sl_raw;
    } else {
        ml = ml_raw * 100.0 / 255.0;
        sl = sl_raw * 100.0 / 255.0;
    }

    selectImage(lab_a_id);
    run("Measure");
    ma_raw  = getResult("Mean",   0);
    sa_raw  = getResult("StdDev", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        ma = ma_raw;
        sa = sa_raw;
    } else {
        ma = ma_raw - 128.0;   // offset shifts mean only
        sa = sa_raw;            // spread unaffected by constant offset
    }

    selectImage(lab_b_id);
    run("Measure");
    mb_raw  = getResult("Mean",   0);
    sb_raw  = getResult("StdDev", 0);
    run("Clear Results");
    if (lab_bd == 32) {
        mb_lab = mb_raw;
        sb_lab = sb_raw;
    } else {
        mb_lab = mb_raw - 128.0;
        sb_lab = sb_raw;
    }

    // -- HSB mean + std (always 8-bit, always needs descaling) --
    selectImage(hsb_id);

    Stack.setSlice(1);
    run("Restore Selection");
    run("Measure");
    mh_raw = getResult("Mean",   0);
    sh_raw = getResult("StdDev", 0);
    mh = mh_raw * 360.0 / 255.0;
    sh = sh_raw * 360.0 / 255.0;
    run("Clear Results");

    Stack.setSlice(2);
    run("Restore Selection");
    run("Measure");
    ms_raw = getResult("Mean",   0);
    ss_raw = getResult("StdDev", 0);
    ms = ms_raw / 255.0;
    ss = ss_raw / 255.0;
    run("Clear Results");

    Stack.setSlice(3);
    run("Restore Selection");
    run("Measure");
    mbr_raw = getResult("Mean",   0);
    sbr_raw = getResult("StdDev", 0);
    mbr = mbr_raw / 255.0;
    sbr = sbr_raw / 255.0;
    run("Clear Results");

    // -- Write row --
    print(f, "" + g + "," + area_val + "," +
             mr  + "," + sr  + "," +
             mg  + "," + sg  + "," +
             mb  + "," + sb  + "," +
             ml  + "," + sl  + "," +
             ma  + "," + sa  + "," +
             mb_lab + "," + sb_lab + "," +
             mh  + "," + sh  + "," +
             ms  + "," + ss  + "," +
             mbr + "," + sbr);
}

// ── Cleanup ───────────────────────────────────────────────────────────────────
File.close(f);
run("Close All");
setBatchMode(false);
eval("script", "System.exit(0);");
