# Graphic-Utility-Radar-Toolkit-V3 BETA 
A graphical interface built on ARM PyART that emulates NCAR's SOLO3 program. 




-----------------------------**USAGE**------------------------------------------

**Controls:**
  Left/Right arrows  — previous/next file in folder
  Up/Down arrows     — higher/lower tilt (sweep)
  Click + drag       — rubber-band zoom into selection
  Right-click panel  — open Parameter & Colors editor for that panel
  Escape             — reset zoom to full range


  **Configs:** 1-5 panels selectable (PANEL COUNT HIGHER THAN 3 WILL HAVE OVERLAPPING ISSUES)


**Data editing:** 
  1. User selected unfolding with PyART region dealias and KDP processing
  2. Unfolding/deglitching brush (adennison2009)
  3. Boundary-guided deletion tool (adennison2009)



**TO-DO:**
1. Add a slider for colormap scaling
2. More file format support (ar2v, ODIM-H5)


**PREREQUISITE PACKAGES:**
1. matplotlib
2. PyART
3. scipy
4. numpy

**Certain code was taken or referenced from these sources:**
lrose-colette, 2025: DeHart, J., Dixon, M., Javornik, B., Bell, M., Cha, T.-Y., DesRosiers, A., & Lee, W.-C. (2025). nsf-lrose/lrose-releases: lrose-colette-20250105 (lrose-colette-20250105). Zenodo. https://doi.org/10.5281/zenodo.14624762

Lee, W.-C., Walther, C., & Oye, R. (2010). National Center for Atmospheric Research NCAR Earth Observing Laboratory EOL DORADE Doppler Radar Exchange Format DORADE Originally. https://www.eol.ucar.edu/sites/default/files/files_live/private/files/field_project/EMEX/DoradeDoc.pdf

Helmus, J.J. & Collis, S.M., (2016). The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software. 4(1), p.e25. DOI: http://doi.org/10.5334/jors.119

