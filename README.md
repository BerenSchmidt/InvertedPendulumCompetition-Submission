This is a much simplified version of the code kept at https://github.com/Shiven110504/InvertedPendulumCompetition.git.
If you want the more full picture including how training occurred, the different models that were trained, and their results go there (specifically Beren's-Branch).

It should be much simpler just to run the files here if you want to see our most consistent model. This model wasn't the one that got the highest balance_count but was more consistent and I haven't seen it drop below 13. 

Here's a screenshot of me running the model to show what you need to input aswell as what I would consider a good run for this model. <img width="1962" height="1127" alt="image" src="https://github.com/user-attachments/assets/16f77c5b-a340-49bd-b13c-5b317d5a06cc" />

I would say generally that it's range is about 15-25 with the occasional higher or lower score.

I made one change to Run_PendulumEnv.py in order to not show the left and right UIs since those block the view when we take videos and I didn't find them super helpful.

The environment.yml file is just there so that you can check that you are running in a similar environment. More details on how I set it up are in the other repo.
