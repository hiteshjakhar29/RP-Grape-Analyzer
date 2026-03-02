// Per-Grape Color Macro
// Automates extraction of color metrics (RGB, Lab, HSB) for EACH individual grape
// Outputs 1 row per grape with: Area, Mean RGB, Mean Lab, Mean HSB
// Based on original macro by Chris Strock, modified for per-grape analysis

// User selects working directory
input = getDirectory("Choose Source Directory ");
list = getFileList(input);

for (i = 0; i < list.length; i++) {
    filename = list[i];
    if (endsWith(filename, ".jpg") || endsWith(filename, ".jpeg") || endsWith(filename, ".png") || endsWith(filename, ".tif") || endsWith(filename, ".tiff")) {
        PerGrapeColor(input, filename);
    }
}

function PerGrapeColor(input, filename) {

    open(input + filename);
    currentImage = getImageID();

    // Open ROI Manager
    run("ROI Manager...");
    roiManager("Reset");

    // Duplicate for mask creation
    run("Duplicate...", "title=Mask");
    selectWindow("Mask");

    // ---- COLOR THRESHOLDING (same as original macro) ----
    // Threshold based on RGB: Blue channel < 136 isolates grapes from white background
    min = newArray(3);
    max = newArray(3);
    filter = newArray(3);
    a = getTitle();
    run("RGB Stack");
    run("Convert Stack to Images");
    selectWindow("Red");   rename("0");
    selectWindow("Green"); rename("1");
    selectWindow("Blue");  rename("2");

    min[0] = 0;   max[0] = 255; filter[0] = "pass";
    min[1] = 0;   max[1] = 255; filter[1] = "pass";
    min[2] = 0;   max[2] = 136; filter[2] = "pass";

    for (k = 0; k < 3; k++) {
        selectWindow("" + k);
        setThreshold(min[k], max[k]);
        run("Convert to Mask");
        if (filter[k] == "stop") run("Invert");
    }

    imageCalculator("AND create", "0", "1");
    imageCalculator("AND create", "Result of 0", "2");
    for (k = 0; k < 3; k++) {
        selectWindow("" + k);
        close();
    }
    selectWindow("Result of 0");
    close();
    selectWindow("Result of Result of 0");
    rename(a);
    // ---- END COLOR THRESHOLDING ----

    // Make binary mask
    setOption("BlackBackground", true);
    run("Make Binary");
    resetThreshold();

    // Morphological cleanup
    run("Fill Holes");
    run("Open", "iterations=2 count=1");

    // Watershed to split touching grapes
    run("Watershed");

    // Analyze Particles to get individual grape ROIs
    // size=5000 filters out noise (grid lines, text) - adjust if needed
    // circularity=0.0-1.0 accepts all shapes (grapes can be elongated)
    run("Analyze Particles...", "size=5000-Infinity circularity=0.00-1.00 show=Nothing clear add");

    n = roiManager("count");

    // Save QC labeled image (shows grape IDs overlaid)
    selectImage(currentImage);
    roiManager("Show All with labels");
    saveAs("Jpeg", input + replace(filename, ".jpg", "") + replace(filename, ".jpeg", "") + "_QC_Labeled.jpg");

    // Print QC info
    print("Image: " + filename + " | Grapes detected: " + n);
    if (n != 36) {
        print("WARNING: Expected 36 grapes, found " + n + " in " + filename);
    }

    // ---- MEASURE RGB ----
    selectImage(currentImage);
    run("Duplicate...", "title=" + filename + "_RGB");
    selectWindow(filename + "_RGB");
    run("Make Composite");
    run("Set Measurements...", "area display mean standard modal min limit decimal=3");
    roiManager("multi-measure measure_all one append");
    close(filename + "_RGB");

    // ---- MEASURE LAB ----
    selectImage(currentImage);
    run("Duplicate...", "title=" + filename + "_LAB");
    selectWindow(filename + "_LAB");
    run("Lab Stack");
    run("Set Measurements...", "area display mean standard modal min limit decimal=3");
    roiManager("multi-measure measure_all one append");
    close(filename + "_LAB");

    // ---- MEASURE HSB ----
    selectImage(currentImage);
    run("Duplicate...", "title=" + filename + "_HSB");
    selectWindow(filename + "_HSB");
    run("HSB Stack");
    run("Set Measurements...", "area display mean standard modal min limit decimal=3");
    roiManager("multi-measure measure_all one append");
    close(filename + "_HSB");

    // Save measurements CSV (long format - Python will reshape to 1 row per grape)
    dir = getDirectory("image");
    outname = replace(filename, ".jpg", "") + replace(filename, ".jpeg", "") + "_PerGrape_Raw.csv";
    saveAs("Measurements", dir + outname);

    // Cleanup
    run("Clear Results");
    roiManager("Reset");
    selectImage(currentImage);
    close("Mask");
    close();
}

print("Done! All images processed.");
