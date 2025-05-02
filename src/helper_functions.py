# Plotting support function

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT as r
from matplotlib.colors import LogNorm
import awkward as ak

# draw matplotlib plots 
def prepare_axis(rangex=None, rangey=None, title='', xlabel='', ylabel=''):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rangex != None:
        ax.set_xlim(rangex)
    if rangey != None:
        ax.set_ylim(rangey)
    return fig, ax

def make_stacked_plots(
        arrays, 
        nbins, 
        range=None, 
        labels=[], 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp'
    ):

    fig, ax = prepare_axis(rangex=range, title=title, xlabel=xlabel, ylabel=ylabel)

    histo, bins = [], []

    for array in arrays:
        h, bin = np.histogram(array, range=range, bins=nbins)
        histo.append(h)
        bins.append(bin)

    for i,h in enumerate(histo):
        hep.histplot(h, bins[i], label=labels[i], ax=ax)
    # Add legend
    ax.legend()

    plt.savefig(f'./figures/{outputname}.png')
    plt.show()

def make_stacked_plots_normalize(
        arrays, 
        nbins, 
        range=None, 
        labels=[], 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        density=True
    ):

    fig, ax = prepare_axis(rangex=range, title=title, xlabel=xlabel, ylabel=ylabel)

    histo, bins = [], []

    for array in arrays:
        h, bin = np.histogram(array, range=range, bins=nbins, density=density)
        histo.append(h)
        bins.append(bin)

    for i,h in enumerate(histo):
        hep.histplot(h, bins[i], label=labels[i], ax=ax)
    # Add legend
    ax.legend()

    plt.savefig(f'./figures/{outputname}.png')
    plt.show()

def make_plots(
        array, 
        nbins, 
        range=None, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        flow=None
    ):

    fig, ax = prepare_axis(rangex=range, title=title, xlabel=xlabel, ylabel=ylabel)

    h, bins = np.histogram(array, range=range, bins=nbins)
    hep.histplot(h, bins, ax=ax, flow=flow)

    # Save plot as a png image
    plt.savefig(f'./figures/{outputname}.png')
    plt.show()

def make_plots_normalize(
        array, 
        nbins, 
        range=None, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp',
        density=True
    ):

    fig, ax = prepare_axis(rangex=range, title=title, xlabel=xlabel, ylabel=ylabel)

    h, bins = np.histogram(array, range=range, bins=nbins, density=density)
    hep.histplot(h, bins, ax=ax)

    # Save plot as a png image
    plt.savefig(f'./figures/{outputname}.png')
    plt.show()

def make_scatter_plot(
        arrayx, 
        arrayy, 
        xrange=None, 
        yrange=None, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp'
    ):

    fig, ax = prepare_axis(title=title, xlabel=xlabel, ylabel=ylabel)

    ax.scatter(arrayx, arrayy)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    # Save plot as a png image
    
    plt.savefig(f'./figures/{outputname}.png')
    plt.show()

def make_2Dhistogram(
        arrayx, 
        nbinsx, 
        rangex, 
        arrayy, 
        nbinsy, 
        rangey, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        scatter=False, 
        cmap='Reds', 
        logscale=False
    ):

    fig, ax = prepare_axis(rangex=rangex, rangey=rangey, title=title, xlabel=xlabel, ylabel=ylabel)
    
    if scatter:
        ax.scatter(arrayx,arrayy)
    else:
        fill, x_edges, y_edges, _ = plt.hist2d(np.array(arrayx), np.array(arrayy),
                                           bins=[nbinsx, nbinsy], range=[rangex, rangey])
        hep.hist2dplot(fill, x_edges, y_edges, ax=ax, cmap=cmap, cbar=True, norm=(LogNorm() if logscale else None))

    return fig, ax, fill, x_edges, y_edges

# draw ROOT plots
def make_2Dhistogram_root(
        arrayx, 
        nbinsx, 
        rangex, 
        arrayy, 
        nbinsy, 
        rangey, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        scatter=False, 
        cmap='Reds', 
        logscale=False, 
        rootfile=None
    ):

    h = r.TH2D(title, title, nbinsx, rangex[0], rangex[1], nbinsy, rangey[0], rangey[1])
    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle(ylabel)

    for i in range(len(arrayx)):
        h.Fill(arrayx[i], arrayy[i])
    h.Draw("colz")
    
    if rootfile:
        rootfile.cd()
        h.Write()

def make_TGraph(
        arrayx, 
        arrayy, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        rangex = None, 
        rangey = None, 
        rootfile=None
    ):

    canvas = r.TCanvas(title, title, 800, 600)
    graph = r.TGraph(len(arrayx), arrayx, arrayy)
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.0)
    graph.Draw("AP")
    graph.GetXaxis().SetLimits(-rangex, rangex)
    graph.GetYaxis().SetRangeUser(-rangey, rangey)
    canvas.Update()
    canvas.Draw()

    if rootfile:
        rootfile.cd()
        canvas.Write()
    
def make_histogram_root(
        array, 
        nbins, 
        hist_range=None, 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        rootfile=None
    ):

    h = r.TH1D(title, title, nbins, hist_range[0], hist_range[1])
    h.GetXaxis().SetTitle(xlabel)
    h.GetYaxis().SetTitle(ylabel)

    array = ak.to_numpy(array)
    for i in range(len(array)):
        h.Fill(array[i])

    if rootfile:
        rootfile.cd()
        h.Write()

def make_stacked_histogram_root(
        arrays, 
        nbins, 
        hist_range=None, 
        labels=[], 
        title='', 
        xlabel='', 
        ylabel='', 
        outputname='temp', 
        rootfile=None
    ):
    
    hists = []
    for i, array in enumerate(arrays):
        h = r.TH1D(title+str(i), title, nbins, hist_range[0], hist_range[1])
        array = ak.to_numpy(array)
        for j in range(len(array)):
            h.Fill(array[j])
        hists.append(h)

    c = r.TCanvas(title, title, 800, 600)
    if rootfile:
        rootfile.cd()
        c.cd()
        hists[0].Draw()
        hists[0].GetXaxis().SetTitle(xlabel)
        hists[0].GetYaxis().SetTitle(ylabel)
        for i in range(1, len(hists)):
            hists[i].Draw("same")
            hists[i].SetLineColor(i+1)

        legende = r.TLegend(0.7, 0.7, 0.9, 0.9)
        for i, label in enumerate(labels):
            legende.AddEntry(hists[i], label, "l")
        legende.Draw()
        c.Write()