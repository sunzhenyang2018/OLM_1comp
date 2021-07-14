# OLM_1comp
single compartment OLM cell model

Reduced from multicompartmodel detailed in the following publication:

  Sekulić V, Yi F, Garrett T, Guet-McCreight A, Lawrence JJ, Skinner FK. 2020. Integration of Within-Cell Experimental Data With Multi-Compartmental Modeling Predicts H-Channel Densities and Distributions in Hippocampal OLM Cells. Frontiers in Cellular Neuroscience. 14. doi:10.3389/fncel.2020.00277.
  
How to run the model:

  the lines in the run file is the premise of the model. Loading init_1comp.hoc grants everything needed for running the model
  
  Current injections:
  
    See use the objects in AP_clamp_1comp.hoc
    
  Recording:
  
    can record directly from h.soma or use functions in Record_1comp
    
  Currentscape:
  
    Pass list of vector objects to plotCurrentscape_6_current in current_visulization
