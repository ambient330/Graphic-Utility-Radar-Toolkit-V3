# Graphic-Utility-Radar-Toolkit-V3 BETA 
A graphical interface built on ARM PyART that emulates NCAR's SOLO3 program. 




-----------------------------USAGE------------------------------------------

Controls:
  Left/Right arrows  — previous/next file in folder
  Up/Down arrows     — higher/lower tilt (sweep)
  Click + drag       — rubber-band zoom into selection
  Right-click panel  — open Parameter & Colors editor for that panel
  Escape             — reset zoom to full range


  Configs: 1-5 panels selectable (PANEL COUNT HIGHER THAN 3 WILL HAVE OVERLAPPING ISSUES)


  Data editing: 
  1. User selected unfolding with PyART region dealias and KDP processing
  2. Unfolding/deglitching brush (adennison2009)
  3. Boundary-guided deletion tool (adennison2009)



TO-DO:
1. Add a slider for colormap scaling
2. More file format support (ar2v, ODIM-H5)


PREREQUISITE PACKAGES:
1. matplotlib
2. PyART
3. scipy
4. numpy
