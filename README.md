This page provides the data associated with the following paper:
"Asynchronous cortical recovery from sleep inertia: region- and frequency-specific dynamics following deep NREM sleep"
in BioRxiv

Chengmi Hu1*, Daisuke Shimane2*, Kiyoshi Nakahara1,2, Masaki Takeda1,2†
1 Graduate School of Engineering, Kochi University of Technology, Kami, Kochi 782-8502,
Japan.
2 Research Center for Brain Communication, Kochi University of Technology, Kami, Kochi

[Data and Code availability]
Figure 2 can be reproduced using supplementary Data 1 and Code 1. Figure 3 can be reproduced using supplementary Data 2 and Code 2. Figure 4 (including Figure S5 and S6) can be reproduced using supplementary Data 3 and Code 3. Figure 5 can be reproduced using supplementary Data 1 and Data 3 and Code 4. Figure 6 can be reproduced using supplementary Data 5 and Code 5.

[Details]
Behavior_data:
This dataset contains the maximum digit span derived from the pre-sleep digit span task (DST) and TrialRTB values.
It also includes averaged psychomotor vigilance task (PVT) measures across all trials for both pre-sleep and post-awakening sessions. Rows correspond to participants.
Running the R script provides descriptive statistics.

RT_data:
This dataset contains reaction time (RT) data collected during the DST, for both pre-sleep and post-awakening conditions. Rows correspond to participants, and columns correspond to trial numbers.

Sleep_characteristics_and_Subjective_measures:
included in behavior data

Sleep_Staging_by_CRNN_last_3min:
This dataset provides sleep stage classifications for the final 3 minutes of sleep for each participant. Sleep stages were estimated at 30-second intervals, resulting in six estimates per participant.

Pupil_size_data:
This dataset contains pupil size measurements collected during the DST. Two conditions are included: pre-sleep and post-awakening. Rows correspond to participants, and columns correspond to trial numbers. (Following the review process, the preprocessing method was updated, and the data were revised accordingly.)

EEG_dataset:
This dataset contains event-related spectral perturbation (ERSP) data derived from the recorded EEG data.

LMM (liner mixed-effects modeling):
This provides the code and data used for the linear mixed-effects modeling.
