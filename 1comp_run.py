import numpy
from neuron import h, gui
import record_1comp as r1
import efel
import matplotlib.pyplot as plt
# import efel
# import glob
# import IPython, os

from currents_visualization import *

### Instantiate Model ###
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514
# cell = h.cell


def preliminary(amps):
    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        vecs = r1.set_up_full_recording()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        fig = plotCurrentscape(vecs[0], vecs[1:])
        plt.savefig("1_comp_plots/preliminary/CS_" + str(amp*1000) + ".png")


def currentscape_run(amps, graph_name):
    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        vecs = r1.set_up_full_recording()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        fig = plotCurrentscape_6_current(vecs[0], vecs[1:])
        plt.savefig("1_comp_plots/" + graph_name + "_" + str(amp*1000) + ".png", dpi=250)


def capacitance_change(factor):
    amps = [0.03]
    origin = h.soma.cm
    h.soma.cm = h.soma.cm / factor

    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        vecs = r1.set_up_full_recording()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        fig = plotCurrentscape(vecs[0], vecs[1:])
        plt.savefig("1_comp_plots/capacitance/cm_div_" + str(factor) + "_" + str(amp*1000) + ".png")

    h.soma.cm = origin


def resistivity_change(factor):
    amps = [0.03]
    origin = h.soma.Ra
    h.soma.Ra = h.soma.Ra / factor

    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        vecs = r1.set_up_full_recording()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        fig = plotCurrentscape(vecs[0], vecs[1:])
        plt.savefig("1_comp_plots/resistance/Ra_div_" + str(factor) + "_" + str(amp*1000) + ".png")

    h.soma.Ra = origin


def fit_passive(model_name):
    amp = -0.12
    types = {0:'v', 1:'ka', 2:'kdrf' , 3:'kdrs' , 4:'kca' , 5:'m' , 6:'l' , 7:'NaD' , 8:'CaT' , 9:'CaL', 10:'h'}
    h.ic_step.amp = amp
    h.ic_hold.amp = DNQX_Hold
    vecs = r1.set_up_full_recording()
    h.run()
    vecs = [numpy.array(i) for i in vecs]
    passive_path = 'passive/'
    fig = figure(figsize=(16, 12))
    for i in range(11):
        plt.subplot(4, 3, i + 1)
        plt.plot(numpy.load(passive_path + types[i] + "_" + str(amp * 1000) + "pA.npy"), color='b')
        plt.plot(vecs[i], color='r')
    plt.savefig("1_comp_plots/" + model_name + '_fit_passive' + ".png", dpi=500)
    plt.close()


def current_overlap(amps, model_name):
    for amp in amps:
        types = {0: 'v', 1: 'ka', 2: 'kdrf', 3: 'kdrs', 4: 'kca', 5: 'm', 6: 'l', 7: 'NaD', 8: 'CaT', 9: 'CaL', 10: 'h'}
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold
        vecs = r1.set_up_full_recording()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        passive_path = 'passive/'
        fig = figure(figsize=(16, 12))
        for i in range(11):
            plt.subplot(4, 3, i + 1)
            plt.plot(numpy.load(passive_path + types[i] + "_" + str(amp * 1000) + "pA.npy"), color='b')
            plt.plot(vecs[i], color='r')
        plt.savefig("1_comp_plots/" + model_name + '_overlay_'+ str(amp*1000) + "pA.png", dpi=500)
        plt.close()


def current_overlap_soma(amps, model_name):
    for amp in amps:
        types = ['v', 'ka', 'kdrf', 'kdrs', 'm', 'l', 'NaS', 'h']
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold
        vecs = r1.record_soma()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        path = 'soma/'
        fig = figure(figsize=(16, 12))
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.plot(numpy.load(path + 'soma_' + types[i] + "_" + str(amp * 1000) + "pA.npy"), color='b')
            plt.plot(vecs[i], color='r', label=types[i])
            plt.legend()
        plt.savefig("1_comp_plots/morph_plots/" + model_name + '_soma_overlay_' + str(amp*1000) + "pA.png", dpi=100)
        plt.close()


def calculate_input_R():
    h.ic_step.amp = -0.12
    h.ic_hold.amp = 0
    voltage = h.Vector()
    voltage.record(h.soma(0.5)._ref_v)
    h.soma.gbar_Ikdrf = 0
    h.soma.gbar_Ika = 0
    h.soma.gna_Nasoma = 0
    h.soma.gbar_IM = 0
    # h.soma.gcalbar_cal = 0
    # h.soma.gbar_cat = 0
    # h.soma.gkbar_kca = 0
    h.soma.gkhbar_Ih = 0
    vecs = r1.set_up_full_recording()
    h.run()
    vecs = [numpy.array(i) for i in vecs]
    voltage = numpy.array(voltage)
    input_R = (voltage[10000] - voltage[30000]) / 0.12
    print(input_R, 'megohms')

    fig = plotCurrentscape(vecs[0], vecs[1:])
    plt.savefig('test.png')


def overlay_cai(amps, model_name):
    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold
        vecs = r1.record_V_cai_ica_dend()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        passive_path = 'passive/'
        plt.plot(vecs[1], color='red', label=model_name)
        plt.plot(numpy.load('passive/soma_cai_trace_'+ str(amp*1000) + 'pA.npy'), color='blue', label='original')
        plt.legend()
        plt.savefig("1_comp_plots/" + model_name + 'cai_overlay_'+ str(amp*1000) + "pA.png", dpi=500)
        plt.close()


def get_sd_area():
    total_area = 0
    for seg in h.soma:
        total_area += h.area(seg.x)


def save_run_data(amps, model_name):
    h.ic_hold.amp = DNQX_Hold
    for amp in amps:
        h.ic_step.amp = amp
        vecs = r1.record_soma()
        h.run()
        vecs = numpy.array([numpy.array(i) for i in vecs])
        numpy.save("1_comp_data/{}/full_record_{}pA.npy".format(model_name, amp*1000), vecs)


def get_resting():
    h.ic_hold.amp = 0
    h.ic_step.amp = 0
    v = h.Vector()
    v.record(h.soma(0.5)._ref_v)
    h.run()
    v = numpy.array(v)
    return numpy.mean(v[10000:])


def currents_output_file(amps):
    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold
        vecs = r1.record_soma()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        with open('curr' + str(amp) + '.txt', 'w') as outfile:
            outfile.write("v, Ika, Ikdrf, Ikdrs, Im, Il, Ina, Ih\n")
            for i in range(len(vecs[0])):
                outfile.write(str(vecs[0][i]))
                outfile.write(", ")
                for j in range(1, len(vecs)):
                    raw = vecs[j][i]
                    convert = raw * 1e3
                    outfile.write(str(convert))
                    outfile.write(", ")
                outfile.write('\n')


def voltage_output_file(amps):
    for amp in amps:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold
        vecs = r1.record_soma()
        h.run()
        vecs = [numpy.array(i) for i in vecs]
        with open('curr' + str(amp) + '.txt', 'w') as outfile:
            for i in range(len(vecs[0])):
                outfile.write(str(vecs[0][i]))
                outfile.write('\n')


def get_area():
    total_area = 0
    for seg in h.soma:
        total_area += h.soma(seg.x).area()
    return total_area


def print_total_conductance(surface_area, conductance_file):
    with open(conductance_file) as infile:
        lines = infile.readlines()
    for line in lines:
        content = line.strip().split("=")
        total_con = surface_area * float(content[1])
        print(content[0] + "=" + str(total_con))


def dt_overlap():
    dt0_1 = np.genfromtxt("dt0.1/curr0.03.txt")
    axis1 = np.linspace(0, 4, 40001)
    axis2 = np.linspace(0, 4, 8000001)
    axis3 = np.linspace(0, 4, 4000001)
    dt0_0005 = np.genfromtxt("dt0.0005/curr0.03.txt")
    dt0_001 = np.genfromtxt("dt0.001/curr0.03.txt")
    plt.figure()
    plt.plot(axis1, dt0_1, label="dt 0.1", color='blue')
    plt.plot(axis2, dt0_0005, label='dt 0.0005', color='red')
    plt.plot(axis3, dt0_001, label='dt 0.001', color='green')
    plt.legend()
    plt.savefig("dt_comparison.png")


def voltage_wt_perturbation(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 100
    h.ic_hold.dur = time
    h.tstop = time


    inter_pert_unit = int(1000 // pert_freq)
    pert_vector = np.zeros(int(time))
    for i in range(150, len(pert_vector), inter_pert_unit):
        pert_vector[i] = pert_amp

    ipert = h.IClamp(h.soma(0.5))
    ipert.delay=0
    ipert.dur=1e9
    timeVec = h.Vector(range(int(time)))
    a = h.Vector(pert_vector)

    a.play(ipert._ref_amp, timeVec, 1)  # The meaning of dt: dt of pert-vector compared to time vector
    b = np.array(a)
    print(sum(b))
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]
    plt.plot(vecs[0])
    plt.savefig("1_comp_plots/pert/hold"+str(hold)+"_pert"+str(pert_amp)+"_freq"+str(pert_freq)+".png", dpi=150)
    # plt.show()


def fit_frequency(hold, time):
    h.ic_hold.amp = hold
    h.ic_hold.dur = time
    h.tstop = time

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    vecs = r1.set_up_full_recording()
    h.run()
    v_trace = np.array(vecs[0])
    tvec = numpy.arange(0, len(v_trace), 1)
    tvec = tvec * h.dt                      # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0]}

    trace['stim_end'] = [time]

    ef_list = ['mean_frequency']
    traces_result = efel.getFeatureValues([trace], ef_list)

    plt.plot(tvec, v_trace)
    plt.show()
    return traces_result


def is3_pert_run(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    stim = h.NetStim()
    stim.interval = 1/pert_freq * 1000       # ms mean time between spikes
    stim.number = int((time-250)*h.dt*1000*pert_freq)        # average number of spikes. convert to ms, then s
    stim.start = 250          # ms start of stim
    stim.noise = 0          # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6          # ms rise time
    syn.tau2 = 12.0         # ms decay time
    syn.e = -87.1           # reversal potential

    netcon = h.NetCon(stim, syn)    # threshold is irrelevant with event-based source
    netcon.weight[0] = pert_amp
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]

    plt.plot(vecs[0][1800:])
    plt.savefig("1_comp_plots/pert/hold" + str(hold) + "_pert" + str(pert_amp) + "_freq" + str(pert_freq) + ".png",
                dpi=150)
    # plt.show()


def plot_frequency():
    run_time = 10000
    h.tstop = run_time
    vecs = r1.set_up_full_recording()
    h.ic_hold.dur = run_time
    h.ic_hold.delay = 0
    frequencies = []

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    for i in range(36):
        h.ic_hold.amp = i * 0.001   # 1 pA = 0.001
        h.run()
        v_trace = np.array(vecs[0])
        tvec = numpy.arange(0, len(v_trace), 1)
        tvec = tvec * h.dt          # mul dt to get 1/1000 second, efel also uses ms
        trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [run_time]}
        ef_list = ['mean_frequency']
        traces_result = efel.getFeatureValues([trace], ef_list)
        if traces_result[0]['mean_frequency'] is None:
            frequencies.append(0)
        else:
            frequencies.append(traces_result[0]['mean_frequency'][0])
    plt.scatter(list(range(36)), frequencies)
    plt.savefig('1_comp_plots/frequencies.png')


def ISI_finder_vitro(hold, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    vecs = r1.record_soma()
    h.run()

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    v_trace = np.array(vecs[0])
    tvec = numpy.arange(0, len(v_trace), 1)
    tvec = tvec * h.dt # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [time]}

    ef_list = ['peak_time']
    spike_times = efel.getFeatureValues([trace], ef_list)
    print(spike_times)
    plt.plot(v_trace)
    plt.show()

# 45.06 pA [{'peak_time': array([ 63.4, 182.1, 317.8, 457.6, 597.7, 737.8, 877.7])}] in ms

def currentscape_wt_pert(hold_amp, runtime):
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    vecs = r1.set_up_full_recording()
    v_vec = vecs[0]

    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime / h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    t0 = spike5 - spike4  # Alex's implementation would be spike4 - spike3

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * t0 + spike4

    for i in range(0, len(pert_time), 20):
        stim.start = pert_time[i]
        h.run()

        plotCurrentscape_6_current(np.array(vecs[0]), np.array(vecs[1:]))
        plt.savefig('1_comp_plots/currscape/' + str(i) + '.png')
        plt.close()


def PRC_vitro(hold_amp, runtime, prcloc, scatterloc):
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    v_vec = r1.set_up_full_recording()[0]

    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime/h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    t0 = spike5 - spike4        # Alex's implementation would be spike4 - spike3

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * t0 + spike4

    phase_shifts = []
    for start_time in pert_time:
        stim.start = start_time
        h.run()
        v_trace = np.array(v_vec)
        trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
        # plt.plot(v_trace)
        # plt.show()
        spike_times = efel.getFeatureValues([trace], ef_list)

        t1 = spike_times[0]['peak_time'][4] - spike4
        percent_shift = (t1-t0)/t0 * 100
        phase_shifts.append(percent_shift)

    plt.plot(percents, phase_shifts)
    plt.savefig(prcloc)     # '1_comp_plots/PRC-vitro.png'
    plt.close()
    plt.scatter(percents, phase_shifts)
    plt.savefig(scatterloc)           # '1_comp_plots/PRC-vitro-scatter.png'

# 31.06 pA [{'peak_time': array([ 779.9, 1609. , 2419. , 3221.4, 4020.3, 4818.2])}] in ms

def PRC_vitro2():
    # set up spiking neuron running 1 second
    h.ic_hold.amp = 0.031
    h.ic_hold.delay = 0
    h.ic_hold.dur = 5000
    h.tstop = 5000

    v_vec = r1.set_up_full_recording()[0]

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    t0 = 4020.3-3221.4
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * (4020.3-3221.4) + 3221.4

    ef_list = ['peak_time']

    phase_shifts = []
    for start_time in pert_time:
        stim.start = start_time
        h.run()
        v_trace = np.array(v_vec)
        tvec = np.array(range(50001))
        tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
        trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [5000]}
        # plt.plot(v_trace)
        # plt.show()
        spike_times = efel.getFeatureValues([trace], ef_list)

        t1 = spike_times[0]['peak_time'][4] - 3221.4
        percent_shift = (t1-t0)/t0 * 100
        phase_shifts.append(percent_shift)

    plt.plot(percents, phase_shifts)
    plt.savefig('1_comp_plots/PRC-vitro2.png')
    plt.close()
    plt.scatter(percents, phase_shifts)
    plt.savefig('1_comp_plots/PRC-vitro-scatter2.png')


def PRCC_vitro(hold_amp, runtime, prcloc): # PRC applied to currents
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    v_vec = r1.set_up_full_recording()[0]
    c_vecs = r1.set_up_full_recording()[1:]
    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime/h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    spike6 = spike_times[0]['peak_time'][5]

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * (spike5 - spike4) + spike4

    phase_shifts = None
    for start_time in pert_time:
        stim.start = start_time
        h.run()
        cur_vecs = np.array([np.array(i) for i in c_vecs])
        # I0 is peak amp from 2nd last spike before pert to pert
        # I1 is peak amp from pert to 2nd spike after pert
        I0_vecs = np.max(abs(cur_vecs[:, int(spike3 / h.dt): int(start_time / h.dt)]), axis=1)  # nx1 array, always
        I1_vecs = np.max(abs(cur_vecs[:, int(start_time / h.dt): int(spike6 / h.dt)]), axis=1)  # nx1 array, always

        single_shift = (I1_vecs-I0_vecs) / I0_vecs * 100

        if phase_shifts is None:
            phase_shifts = single_shift
        else:
            # in the end, phase_shifts will be size: 100 x num of curs. 1st col is 1st current
            phase_shifts = np.vstack((phase_shifts, single_shift))

    # plt.plot(np.tile(np.array(range(101)), (phase_shifts.shape[1], 1)), phase_shifts.T)
    labels = ['ka', 'kdrf', 'm', 'l', 'na', 'h']
    for i in range(phase_shifts.shape[1]):
        plt.plot(np.array(range(101)), phase_shifts[:, i], label=labels[i])
    plt.legend()
    plt.savefig(prcloc)     # '1_comp_plots/PRCC-vitro.png'


def pyr_pert_run(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    stim = h.NetStim()
    stim.interval = 1 / pert_freq * 1000  # ms mean time between spikes
    stim.number = int((time) * h.dt * 1000 * pert_freq)  # average number of spikes. convert to ms, then s
    stim.start = 0 # ms start of stim
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 2.4  # ms rise time
    syn.tau2 = 12.7  # ms decay time
    syn.e = 0  # reversal potential from f1000

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = pert_amp
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]

    plt.plot(vecs[0][90000:])
    plt.savefig("1_comp_plots/pert/PYR_hold" + str(hold) + "_pert" + str(pert_amp) + "_freq" + str(pert_freq) + ".png",
                dpi=150)
    # plt.show()


def txt_runs():
    for amp in [0.03, 0.06, 0.09, -0.12]:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        v_vec = r1.set_up_full_recording()[0]
        h.run()
        v_vec = np.array(v_vec)
        np.savetxt(str(amp) + "pA.txt", v_vec)


# amps = np.array([0, 0.03, 0.06, 0.09, -0.12])
# amps = np.array([0.03])




# fit_passive('morph10')
# currentscape_run(amps, 'm1053s_cap1_f14_na')
# currentscape_run(amps, 'morph10_5_3_s')
# current_overlap(amps, 'morph10_5_3_S')
# current_overlap_soma(amps, 'morph10_5_3_S')
# calculate_input_R()
# overlay_cai('morph11')
# save_run_data(amps, "morph10_5_3_S")
# print(get_resting())
# currents_output_file(amps)
# voltage_output_file(amps)
# print(get_area())
# print("1-comp:")
# print_total_conductance(get_area(), "conductance.txt")
# print("soma:")
# print_total_conductance(7650.910291111633, "conductance.txt")
# print("dend:")
# print_total_conductance(21727.226670024476, "conductance.txt")
# dt_overlap()

# voltage_wt_perturbation(0.04506, -0.3, 5, 1100)
# print(fit_frequency(0.1379, 10000))
# is3_pert_run(0.04506, 0.0054, 5, 1000)

# ISI_finder_vitro(0.031, 5000)
# PRC_vitro(0.03075, 10000, '1_comp_plots/PRC-vitro.png', '1_comp_plots/PRC-vitro-scatter.png')
# PRC_vitro(0.03705, 10000, '1_comp_plots/PRC-vitro4hz.png', '1_comp_plots/PRC-vitro-scatter4hz.png')
# PRC_vitro2()

# PRCC_vitro(0.03075, 10000, '1_comp_plots/PRCC-vitro1hz.png')
# currentscape_wt_pert(0.03075, 10000)

# pyr_pert_run(0.04506, 0.006, 5, 10000)

txt_runs()
