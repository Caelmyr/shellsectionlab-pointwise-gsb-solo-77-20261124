# Shell Section Lab

Compare registered photographs of one cut shell surface. Open the synthetic demonstration to try two polarization views and a macro image. For your own photographs, create a run with the real frame dimensions, upload a PNG, JPEG or TIFF and mark the same four physical corners in each photo. Then mark the samples and take a reading. Reopen saved readings to inspect or download their overlay, values and original photographs.

The analysis desk provides section tone, matrix and inclusion contrast, inclusion fraction, lamina spacing, pore candidates, line pore intercept, growth-boundary transition, landmark offsets, a paired landmark adjustment, polarization response, paired visibility, local alteration, registered color difference and corrected registered color difference. The linked image preparation room supports independent photo preparation before import.

The paired landmark adjustment fits two corrections from matched landmarks in the already calibrated millimetre plane: a rigid rotation/translation fit that keeps the registered millimetre scale, and a uniform-similarity fit that adds one isotropic scale factor. Both are tested against the limits you enter; the rigid fit is adopted whenever it qualifies (to preserve millimetre distances), otherwise the similarity fit is adopted when it meets both the residual and scale limits, and when neither qualifies the record explains which limits were exceeded. A corrected registered color difference then samples the comparison photograph through the adopted transform, so every point in the reference window is compared against its corrected counterpart, and positions without usable comparison pixels are reported and drawn as missing evidence. The four-corner registration stays in force for every other reading.

The displayed fractions and pore candidates come from image evidence in the selected marks. Color distances use display-referred Lab. The polarization response is relative to the two controls; it does not determine material composition. The calibration assumes a single flat plane and uses at most 1,300 pixels along its long side. Fine features below the resolved image scale should remain uncertain. The sample photographs are generated illustrations.

## Run locally

From the project directory, run `./start.sh` and open `http://127.0.0.1:19992/`. Docker Compose keeps the sample library and preparation projects in the named volume `section-library`. Run `./start.sh --down` to stop the service. To change the loopback port, set `SECTIONLAB_PORT` when starting the project.

For a native session, install the exact packages in `requirements.txt` into an isolated Python environment, then run `HOST=127.0.0.1 PORT=19992 python -m sectionlab`. Set `SECTIONLAB_DATA` to an independent data path if you run more than one instance.
