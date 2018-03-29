# AC Dataset

We use the arg-microtexts corpus of Andreas Peldszus and Manfred Stede: [An annotated corpus of argumentative microtexts](https://github.com/peldszus/arg-microtexts).

> Andreas Peldszus, Manfred Stede. An annotated corpus of argumentative microtexts. First European Conference on Argumentation: Argumentation and Reasoned Action, Portugal, Lisbon, June 2015.

We changed the data format for our purposes and distribute it in this folder. Our derivations are under the same [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) as the original.

## Files 

  * amClaims.en: english sentences
  * amClaims.de: german translation of the english sentences (human translations)
  * amClaims.fr: machine translated sentences (from english to french)
  * amClaims.MT-de: machine translated sentences (from english to german). We used this file to measure ranking differences between models on machine translations and human translations
  * amClaims.labels: contains labels for all sentences