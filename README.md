# PIVOT
## Planetary Interactions: Variations Of Timing

Performs a Transit Timing Variation analysis for a planet given parameters and lightcurve data.

Installation Steps:
<ol>
    <li> Run <code>pip install -r requirements.txt</code> to download correct dependencies</li> 
    <li> For Kepler or K2 planets: use <code>ttv.download_lightkurve(system_name, instrument)</code> given in ttv.py to download lightcurve data</li>
    <li>Run TTV analysis using <code>ttv.ttv_algo</code>.</li>
</ol>

Please check example.ipynb for test planet run.

If you find this code useful in your analysis, please cite Lopez Murillo et al. 2026  <br>
DOI: <a href="https://iopscience.iop.org/article/10.3847/1538-3881/ae231a">10.3847/1538-3881/ae231a</a>
