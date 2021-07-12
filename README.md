# OLM_1comp
single compartment OLM cell model

Reduced from multicompartmodel detailed in the following publication:
  Sekulić V, Lawrence JJ, Skinner FK. 2014. Using Multi-Compartment Ensemble Modeling As an Investigative Tool of Spatially Distributed Biophysical Balances: Application to  Hippocampal Oriens-Lacunosum/Moleculare (O-LM) Cells. PLOS ONE. 9(10):e106567. doi:10.1371/journal.pone.0106567.
  
How to run the model:
  the lines in the run file is the premise of the model. Loading init_1comp.hoc grants everything needed for running the model
  Current injections:
    See use the objects in AP_clamp_1comp.hoc
  Recording:
    can record directly from h.soma or use functions in Record_1comp
  Currentscape:
    Pass list of vector objects to plotCurrentscape_6_current in current_visulization
