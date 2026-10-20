For your setup, I would start with a global TCN (or DeepTCN) as the primary neural model, then test a Temporal Fusion Transformer (TFT) if you need multi-step forecasts, uncertainty, or richer handling of static plant metadata. Your data are a classic panel time series: many companies observed over time, each with its own production history, weather, and capacity.
lutpub.lut
+2

Best first choice: global TCN
Use one shared neural network for all companies, not one separate model per company. This is called a global or cross-learning model: it learns common solar-production dynamics—day/night cycles, seasonal patterns, irradiance-to-power response, temperature effects—while company identity and plant characteristics distinguish individual sites.
lutpub.lut
+1

A TCN is especially suitable because:

It models the ordered history of production and weather directly through causal 1D convolutions.

Dilated convolutions can capture both short-term changes and longer daily/weekly/seasonal patterns.

It is usually easier and faster to train than recurrent models, because convolutional computations parallelize across time.

A global TCN can learn from all companies simultaneously, which helps smaller companies with limited history.
lutpub.lut
+2

For next-time-step forecasting, a TCN is a very strong, practical choice. Research on PV systems also finds that at hourly resolution, the TCN architecture itself can provide the dominant accuracy gain, while weather covariates become especially important at finer resolutions such as 15 minutes.
arxiv

Why not an MLP alone?
An MLP is useful, but it should not be your only model if the recent production trajectory matters. It sees a fixed feature vector, so it does not naturally understand that yesterday’s production, this morning’s irradiance, and the last few hours of temperature form a sequence.

You could still use an MLP as a strong baseline:

Engineer lags: production at 
𝑡
−
1
t−1, 
𝑡
−
24
t−24, 
𝑡
−
168
t−168, rolling means, and weather lags.

Add current and forecast weather: solar irradiance, temperature, cloud cover, wind, humidity.

Add capacity and availability: installed capacity, available capacity, outage/maintenance flags.

Add calendar features: hour, day of week, month, holiday, sunrise/sunset-related features.

Add company identity as an embedding, plus static features such as location, orientation, tilt, technology, and installed capacity.

This MLP baseline is valuable because it is fast, interpretable through feature importance tools, and tells you how much performance comes simply from good features.
journals.plos

Strong alternative: TFT
If you forecast more than one step ahead—for example, the next 24 hours—or need a production-ready model that handles heterogeneous inputs, choose TFT. It is designed for multi-horizon forecasting with:

Past time-varying inputs: historical production, irradiance, temperature, availability.

Known future inputs: weather forecasts, calendar variables, planned maintenance/outages.

Static covariates: company ID, location, installed capacity, panel technology, tilt/orientation.

Variable selection and attention: helps identify which inputs matter and can suppress irrelevant features.
link.springer
+2

TFT has been applied successfully to day-ahead PV forecasting across multiple facilities and has outperformed ARIMA, LSTM, MLP, and XGBoost in at least one multi-site study. It is more complex than a TCN, so I would use it after establishing a solid TCN baseline—not as the first experiment.
journals.plos

Model choice by use case
Your requirement	Recommended model	Why
One-step-ahead forecast, strong baseline	Global TCN	Learns sequence patterns, trains efficiently, works well with panel data. 
arxiv
+2
Need uncertainty intervals	DeepAR or quantile TCN/TFT	DeepAR is explicitly probabilistic and learns globally across related series. 
arxiv
+1
Multi-step/day-ahead forecast	TFT	Handles known future weather, static plant metadata, and long-range dependencies. 
journals.plos
+2
Small dataset or need interpretability	MLP or gradient boosting	Simple, robust, and effective with carefully engineered lag/weather features. 
journals.plos
Best production system	Ensemble: TCN + gradient boosting/MLP	Different model families reduce the risk that one architecture misses an important pattern. 
journals.plos
+1
Recommended data setup
Your target should normally be normalized production, for example:

𝑦
𝑖
,
𝑡
=
production
𝑖
,
𝑡
available capacity
𝑖
,
𝑡
y 
i,t
​
 = 
available capacity 
i,t
​
 
production 
i,t
​
 
​
 
This makes companies of different sizes comparable and prevents a large plant from dominating the loss function. Forecast this ratio, then multiply the prediction by the company’s known available capacity at the target time.

Use these input groups:

Target history: lagged production or capacity-normalized production.

Weather history: irradiance, temperature, cloud cover, wind, humidity.

Future weather forecast: irradiance and temperature for the target timestamp; these are known future covariates in operational forecasting.

Plant state: available capacity, outage flag, maintenance flag, curtailment flag.

Static plant features: company ID embedding, location, installed capacity, technology, tilt, azimuth.

Calendar: hour, day of week, month, holiday, daylight duration, solar elevation or clear-sky irradiance.

Avoid leakage: never use the target company’s production at the forecast time, and ensure weather values for the future are genuinely forecasts available at prediction time—not observed weather measured afterward.

Practical modeling plan
Build a global MLP baseline with lag, rolling, weather, calendar, capacity, and company-embedding features.

Build a global TCN using a lookback window such as 24, 48, or 168 periods, depending on whether your data are hourly, 15-minute, or daily.

Add future known covariates—especially weather forecasts and calendar features—to the TCN.

Compare against the MLP using rolling-origin time-series cross-validation.

If multi-step forecasting or interpretability matters, train a TFT and compare it with the TCN.

If the best models are close, blend their predictions; a diverse ensemble is often more robust than relying on one deep-learning architecture.
journals.plos
+1

My concrete recommendation
For your exact task, I would choose:

Global TCN with company embeddings, weather/capacity covariates, and future weather inputs as the main neural model.

Then benchmark it against an MLP baseline and, if you need day-ahead or probabilistic forecasts, a TFT. This approach uses the panel structure efficiently, respects causality, captures solar-specific temporal patterns, and avoids the unnecessary complexity of training a separate network for every company.