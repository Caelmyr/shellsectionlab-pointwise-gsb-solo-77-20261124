# Shell Section Lab

Compare registered photographs of one cut shell surface. Open the synthetic demonstration to try two polarization views and a macro image. For your own photographs, create a run with the real frame dimensions, upload a PNG, JPEG or TIFF and mark the same four physical corners in each photo. Then mark the samples and take a reading. Reopen saved readings to inspect or download their overlay, values and original photographs.

The analysis desk provides section tone, matrix and inclusion contrast, inclusion fraction, lamina spacing, pore candidates, paired pore matching, line pore intercept, paired line pore intercept, growth-boundary transition, landmark offsets, polarization response, paired visibility, local alteration and registered color difference. The linked image preparation room supports independent photo preparation before import.

The paired pore map matches the same physical dark pores between two registered photographs inside one inspection window. A pair must satisfy both the maximum pore-centre separation and the minimum shared-area criteria, and each dark pore is used at most once. When several legal matchings exist, the adopted one maximizes the number of pairs, then the total shared area, then minimizes total centre separation; only this matching is stored in the immutable reading. Unmatched dark spots are recorded as missing evidence when the other photograph does not observe that location, or as single-photo dark spots when it does. The paired line pore intercept references a saved paired pore map and reports which adopted pores a new survey line crosses and each crossing length, listing single-photo spots and missing evidence separately. Saving another ordinary pore map never changes a stored pairing.

The displayed fractions and pore candidates come from image evidence in the selected marks. Color distances use display-referred Lab. The polarization response is relative to the two controls; it does not determine material composition. The calibration assumes a single flat plane and uses at most 1,300 pixels along its long side. Fine features below the resolved image scale should remain uncertain. The sample photographs are generated illustrations.

## Run locally

From the project directory, run `./start.sh` and open `http://127.0.0.1:19992/`. Docker Compose keeps the sample library and preparation projects in the named volume `section-library`. Run `./start.sh --down` to stop the service. To change the loopback port, set `SECTIONLAB_PORT` when starting the project.

For a native session, install the exact packages in `requirements.txt` into an isolated Python environment, then run `HOST=127.0.0.1 PORT=19992 python -m sectionlab`. Set `SECTIONLAB_DATA` to an independent data path if you run more than one instance.
