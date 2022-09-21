Here we again sample the ABCD diffusion imaging data, but this time with the objective of conducting an analysis of a various registration technqiues.

Here is how the sample ends up being generated:
- Take the set of subjects and drop the few of them that went to different study sites or used different scanner models between their two scans.
- Take the remaining set of subjects and sample 0.03 of them, stratified by baseline age, gender, study site, and scanner model.
- Mark 0.2 of the selected subjects as being part of the *test set*, which will be used for final evaluations only and not for any hyperparameter tuning.
- From that marked test set, make sure to remove any subject that happened to show up in the sample that was used for the previous exploration 01.0_abcd_sample/sampled_subjectkeys.csv. This ensures that what was learned about hyperparameters in the earlier exploration doesn't get bound up with the data used for final evaluation and publication of results.
- For each subject in the test set and the nontest set, randomly pick between using their baseline image or their 2 year follow up image. We never take both time points for the same subject.