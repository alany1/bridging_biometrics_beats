# 6.8611 Final Project

Alan Yu Evan Liu Michael Peng

watch -> LM (few shot or feature generator) -> "watch target" t_w

two options to generate initial recommendations:

1. t_w + predict genre -> SP -> recs
2. t_w + my top k songs -> SP -> recs

how to adjust to user feedback? input x

x -> LM -> which audio components to adjust
small shift in each coordinate (hyperparameter)
- database, for each user save 
                each context vector for each different context

can also ask user to rank on a scale from 1 to 10
