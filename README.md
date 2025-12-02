This is a much simplified version of the code kept at https://github.com/Shiven110504/InvertedPendulumCompetition.git.

This includes only one of the models that we trained. It's not the best in all aspects but it is all around good.

If you want the more full picture including how training occurred, the different models that were trained, and their results go there (specifically Beren's-Branch).

It should be much simpler just to run the files here if you want to see our most consistent model. This model wasn't the one that got the highest balance_count but was more consistent and I've rarely seen it drop below 15. 

The environment.yml file is just there so that you can check that you are running in a similar environment. More details on how I set it up are in the other repo. Once it is setup feel free to run Run_PendulumEnv.py.

Here's a screenshot of the output after running the model to show what you need to input to make sure the agent is working.
<img width="1962" height="1127" alt="image" src="https://github.com/user-attachments/assets/16f77c5b-a340-49bd-b13c-5b317d5a06cc" />

I would consider this a good run for this model. From our models we generally get a 10-40 range averaging around 20. This one averages around 22 but I haven't seen it score as high as some of the others.

I would say generally that it's range is about 15-29 with the occasional higher or lower score.

I made one change to Run_PendulumEnv.py in order to not show the left and right UIs since those block the view when we take videos and I didn't find them super helpful.

If you are having trouble with this then going to the repo may have a more thurough readme explanation.
