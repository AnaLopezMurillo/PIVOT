# PIVOT
## Planetary Interactions: Variations Of Timing

Performs a Transit Timing Variation analysis for a planet given parameters and lightcurve data.

Installation Steps:
<ol>
    <li> Run <code>pip install -r requirements.txt</code> to download correct dependencies</li> 
    <li> For Kepler or K2 planets: use <code>ttv.download_lightkurve(system_name, instrument)</code> given in ttv.py to download lightcurve data</li>
    <li>Run TTV analysis using ttv.ttv_algo.</li>
</ol>

Please check example.ipynb for test planet run.