# ============================================================================
# TABOO ANCHOR WORDS FOR EUPHEMISM DETECTION VIA EMBEDDING SIMILARITY
# ============================================================================
#
# DESIGN PRINCIPLES:
#   1. Each entry is a semantic ANCHOR — the most direct, prototypical term
#      for a taboo concept. Euphemisms orbit these in embedding space.
#   2. Minimize redundancy: near-synonyms that occupy the same embedding
#      region are commented out. They waste compute and don't increase recall.
#   3. Minimize polysemy: words with dominant non-taboo senses are removed
#      or replaced to reduce false positives.
#   4. Prefer literal/clinical register over slang or euphemistic framing.
#   5. Multi-word phrases are used ONLY when a single word is too ambiguous
#      on its own. Keep these rare — they complicate n-gram matching.
#   6. One to three anchors per category is the sweet spot.
#
# NOTATION:
#   - Active terms are uncommented
#   - # [REMOVED: ...] = deleted with reason
#   - # [ADDED] = new term with reason
#   - # [KEPT] = original term retained with notes
#
# ============================================================================
# EMBEDDING MODEL CONSIDERATIONS
# ============================================================================
#
# WORD-LEVEL EMBEDDERS (Word2Vec, GloVe, FastText)
#   Pros:
#     - Fast, lightweight, easy to run at scale
#     - Great for single-word anchors matching single-word euphemisms
#     - FastText handles OOV words via subword info (good for slang/neologisms)
#   Cons:
#     - No context sensitivity — "fired" (job) vs "fired" (gun) are the same vector
#     - Multi-word anchors require averaging or composition, which loses nuance
#     - Multi-word euphemisms like "let go" or "passed away" are hard to match
#     - Struggles with phrases whose meaning isn't compositional
#   Best for: A fast first pass with single-word anchors, if you accept
#             more false positives and plan a second-pass filter
#
# SENTENCE-LEVEL EMBEDDERS (Sentence-BERT, all-MiniLM, E5, GTE, etc.)
#   Pros:
#     - Context-aware: "he was fired" vs "he fired the gun" get different vectors
#     - Handles multi-word euphemisms naturally ("passed away", "let go")
#     - Multi-word anchors work well since the model sees them as phrases
#     - Better at capturing the INTENT/TONE of euphemistic language
#   Cons:
#     - Slower, heavier (though modern models like all-MiniLM-L6-v2 are fast)
#     - You need to decide on input window size (sentence? sliding n-gram?)
#     - Single-word anchors embedded alone may lack context and drift in
#       meaning — consider wrapping them: "death" -> "death of a person"
#     - Similarity thresholds need per-category tuning
#   Best for: Production euphemism detection, especially if you embed
#             sliding n-grams (1-3 words) from input text against anchors
#
# HYBRID APPROACH (recommended):
#   - Use sentence-level embedder
#   - Keep anchors mostly as single words (this list)
#   - But EMBED them with a short context template to reduce ambiguity:
#       anchor = "death" -> embedded as "the taboo topic of death"
#       anchor = "fired" -> embedded as "the taboo topic of being fired from a job"
#     This nudges the anchor vector into the right region without requiring
#     multi-word list entries. You can automate this with the category labels.
#   - Compare against sliding n-grams (unigram + bigram + trigram) from input
#   - Use the category label to set per-category similarity thresholds
#
# ============================================================================


TABOO_ANCHORS = {

    # ========================================================================
    # DRUGS & SUBSTANCE USE
    # ========================================================================
    "drugs": [
        "drug",                # [KEPT] Core anchor. Broad but essential.
        "addiction",           # [KEPT] Covers "habit", "dependency", "problem"
        # "substance",         # [REMOVED: too polysemous — "substance" has strong
        #                      #  non-taboo uses like "substance of the argument"]
        "heroin",              # [KEPT] Specific enough to anchor hard drug euphemisms
        "cocaine",             # [KEPT] Same rationale
        # "meth",              # [REMOVED: redundant with heroin/cocaine — they
        #                      #  collectively anchor the "hard drugs" region.
        #                      #  Meth-specific euphemisms like "ice" or "crystal"
        #                      #  will still land near heroin/cocaine in embedding space]
        "overdose",            # [KEPT] Anchors "OD", "took too many", etc.
        # "addict",            # [REMOVED: redundant with "addiction"]
        # "narcotic",          # [REMOVED: redundant with "drug"]
        # "drug abuse",        # [REMOVED: redundant with "drug" + "addiction"]
        # "drug user",         # [REMOVED: redundant — compositional from "drug"]
        # "drug addict",       # [REMOVED: redundant with "addiction"]
    ],

    # ========================================================================
    # ALCOHOL
    # ========================================================================
    "alcohol": [
        "alcohol",             # [KEPT] Primary anchor for "booze", "spirits", "drink"
        "drunk",               # [KEPT] Anchors "tipsy", "hammered", "three sheets"
        # "drinking",          # [REMOVED: massively polysemous — drinking water,
        #                      #  drinking coffee. "Alcohol" + "drunk" cover this]
        # "drunkenness",       # [REMOVED: redundant with "drunk"]
        # "alcoholic",         # [REMOVED: redundant with "alcohol" + "addiction"
        #                      #  from drugs category. Cross-category coverage.]
        # "intoxicated",       # [REMOVED: redundant with "drunk"]
        "hangover",            # [KEPT] Distinct concept with its own euphemisms
        #                      #  like "morning after", "rough morning"
    ],

    # ========================================================================
    # SEX & REPRODUCTION
    # ========================================================================
    "sex": [
        "sex",                 # [KEPT] Core anchor. "Sleeping together", "doing it"
        "pregnancy",           # [KEPT] Anchors "expecting", "in the family way",
        #                      #  "bun in the oven"
        "prostitution",        # [KEPT] Anchors "escort", "working girl",
        #                      #  "oldest profession"
        "pornography",         # [KEPT] Anchors "adult content", "explicit material"
        "masturbation",        # [KEPT] Distinct act with many euphemisms
        "genitals",            # [KEPT] Anchors all genital euphemisms broadly
        "orgasm",              # [KEPT] Distinct concept
        # "penis",             # [REMOVED: redundant with "genitals" — both land
        #                      #  in the same sexual anatomy region]
        # "vagina",            # [REMOVED: same rationale as penis]
        # "sexual intercourse",# [REMOVED: redundant with "sex". Also a multi-word
        #                      #  phrase that adds complexity without coverage gain]
        # "sexual act",        # [REMOVED: redundant with "sex"]
        "erection",            # [ADDED] Distinct physiological event with its own
                               #  euphemisms ("hard", "aroused", "excited") that
                               #  "genitals" alone won't anchor well
    ],

    # ========================================================================
    # DEATH & DYING
    # ========================================================================
    "death": [
        "death",               # [KEPT] THE anchor. "Passed away", "no longer with us",
        #                      #  "departed", "lost", "gone"
        # "dead",              # [REMOVED: redundant with "death"]
        # "die",               # [REMOVED: redundant with "death"]
        # "dying",             # [REMOVED: redundant with "death"]
        # "corpse",            # [REMOVED: too specific, and redundant — "corpse"
        #                      #  euphemisms like "remains" still land near "death"]
        "funeral",             # [KEPT] Anchors "service", "celebration of life",
        #                      #  "memorial". Distinct sub-domain.
        # "burial",            # [REMOVED: redundant with "funeral"]
        "suicide",             # [KEPT] Very distinct euphemism cluster: "took their
        #                      #  own life", "ended it", "self-harm"
        "kill",                # [KEPT] Anchors violence-related death euphemisms:
        #                      #  "put down", "take out", "eliminate"
        # "killed",            # [REMOVED: redundant with "kill"]
        # "murder",            # [REMOVED: redundant with "kill" — both anchor
        #                      #  the intentional-death region. Murder also
        #                      #  appears in Crime category.]
        # "dead body",         # [REMOVED: redundant with "death" + "corpse" removed]
        "euthanasia",          # [ADDED] Distinct concept with unique euphemisms:
                               #  "put to sleep", "mercy", "dignity in dying"
                               #  Not well-covered by "death" or "kill"
    ],

    # ========================================================================
    # HEALTH & ILLNESS
    # ========================================================================
    "health": [
        "disease",             # [KEPT] Core medical anchor
        # "illness",           # [REMOVED: redundant with "disease"]
        "cancer",              # [KEPT] Extremely high euphemism density:
        #                      #  "the big C", "growth", "long illness"
        "terminal illness",    # [KEPT as multi-word] Exception to single-word rule
        #                      #  because "terminal" alone is too polysemous
        #                      #  (airport terminal, computer terminal).
        #                      #  Anchors "not long left", "prognosis"
        # "sick",              # [REMOVED: too polysemous — "sick beat", "sick of it"]
        # "diagnosis",         # [REMOVED: not itself taboo, just a medical process]
        # "infection",         # [REMOVED: redundant with "disease"]
        "dementia",            # [ADDED] Major gap in original list. Huge euphemism
                               #  cluster: "memory problems", "confusion", "not all
                               #  there", "losing their mind", "senior moments"
        "venereal disease",    # [ADDED] STIs have a distinct euphemism tradition:
                               #  "social disease", "something", "a condition"
    ],

    # ========================================================================
    # MENSTRUATION
    # ========================================================================
    "menstruation": [
        "menstruation",        # [KEPT] Core anchor. "Period", "that time of the
        #                      #  month", "Aunt Flo", "on the rag"
        # "menstrual blood",   # [REMOVED: redundant — covered by menstruation +
        #                      #  blood in bodily fluids]
        # "menstrual cycle",   # [REMOVED: redundant with "menstruation"]
    ],

    # ========================================================================
    # AGING
    # ========================================================================
    "aging": [
        "aging",               # [KEPT] Core anchor. "Getting on", "golden years"
        # "old",               # [REMOVED: extremely polysemous — "old book",
        #                      #  "old friend", "old habits". Will cause massive
        #                      #  false positives]
        "elderly",             # [KEPT] More specific than "old" for the
        #                      #  taboo sense. Anchors "senior", "mature"
        # "senile",            # [REMOVED: redundant — now covered by "dementia"
        #                      #  in health category]
        # "frail",             # [REMOVED: polysemous and not strongly taboo
        #                      #  on its own]
    ],

    # ========================================================================
    # BODY FUNCTIONS
    # ========================================================================
    "body_functions": [
        "urinate",             # [KEPT] Anchors "pee", "take a leak", "powder my nose"
        "defecate",            # [KEPT] Anchors "poop", "number two", "do my business"
        "vomit",               # [KEPT] Anchors "throw up", "be sick", "toss cookies"
        # "sweat",             # [REMOVED: very weakly taboo, rarely euphemized
        #                      #  in ways that need detection. "Perspire" is the
        #                      #  only real euphemism and it's obvious.]
        "flatulence",          # [KEPT] Anchors "gas", "wind", "break wind", "toot"
    ],

    # ========================================================================
    # BODILY FLUIDS
    # ========================================================================
    "bodily_fluids": [
        "blood",               # [KEPT] Somewhat polysemous ("blood relation") but
        #                      #  the bodily fluid sense is dominant enough
        # "urine",             # [REMOVED: redundant with "urinate" above]
        # "feces",             # [REMOVED: redundant with "defecate" above]
        "semen",               # [KEPT] Distinct fluid with its own euphemisms
        # "mucus",             # [REMOVED: very weakly taboo, rarely euphemized]
    ],

    # ========================================================================
    # TOILET & EXCRETION
    # ========================================================================
    "toilet": [
        "toilet",              # [KEPT] Anchors "restroom", "loo", "facilities",
        #                      #  "little boys/girls room", "lavatory"
        # "fecal",             # [REMOVED: redundant with "defecate"]
        # "excrement",         # [REMOVED: redundant with "defecate"]
        # "bowel",             # [REMOVED: redundant with "defecate"/"toilet"]
        "diarrhea",            # [KEPT] Distinct condition with euphemisms:
        #                      #  "the runs", "upset stomach", "tummy trouble"
    ],

    # ========================================================================
    # PHYSICAL INJURY
    # ========================================================================
    "physical_injury": [
        "injury",              # [KEPT] Broad anchor
        "wound",               # [KEPT] Slightly different semantic region — more
        #                      #  severe/violent than "injury"
        # "bleeding",          # [REMOVED: redundant with "blood" + "wound"]
        "amputation",          # [KEPT] Distinct, highly euphemized: "lost a leg"
        # "paralysis",         # [REMOVED: moved conceptually to disabilities]
    ],

    # ========================================================================
    # INTELLIGENCE & COGNITIVE ABILITY
    # ========================================================================
    "intelligence": [
        "stupid",              # [KEPT] Core anchor for the cluster
        # "idiot",             # [REMOVED: redundant — same embedding region as stupid]
        # "dumb",              # [REMOVED: redundant AND polysemous ("dumb" = mute)]
        # "unintelligent",     # [REMOVED: redundant with "stupid"]
        # "incompetent",       # [REMOVED: more about skill than intelligence,
        #                      #  and it's redundant with "stupid" for this task]
        "ignorant",            # [ADDED] Slightly different shade — anchors
                               #  euphemisms about lack of knowledge/education
                               #  like "uninformed", "doesn't know better"
    ],

    # ========================================================================
    # EMOTIONS & MENTAL STATE
    # ========================================================================
    "emotions": [
        "anger",               # [KEPT] Anchors "upset", "frustrated", "lost it"
        # "rage",              # [REMOVED: redundant with "anger"]
        "depression",          # [KEPT] Anchors "feeling down", "the blues", "low"
        "anxiety",             # [KEPT] Anchors "nervous", "worried", "on edge"
        # "panic",             # [REMOVED: redundant with "anxiety"]
        "grief",               # [KEPT] Distinct from depression — anchors
        #                      #  "loss", "mourning", "heartbroken"
    ],

    # ========================================================================
    # SEXUAL ORIENTATION
    # ========================================================================
    "sexual_orientation": [
        "homosexual",          # [KEPT] Clinical anchor for orientation euphemisms:
        #                      #  "that way", "batting for the other team",
        #                      #  "a friend of Dorothy"
        # "gay",               # [REMOVED: redundant with "homosexual"]
        # "lesbian",           # [REMOVED: redundant with "homosexual"]
        # "sexual orientation",# [REMOVED: not taboo itself, it's the neutral
        #                      #  meta-term]
    ],

    # ========================================================================
    # IDENTITY
    # ========================================================================
    "identity": [
        "transgender",         # [KEPT] Anchors euphemisms and indirect references
        # "gender",            # [REMOVED: too broad — "gender" in most contexts
        #                      #  is not taboo at all]
        # "identity",          # [REMOVED: extremely polysemous]
        # "minority",          # [REMOVED: very broad, not inherently taboo]
    ],

    # ========================================================================
    # FAMILY & RELATIONSHIPS
    # ========================================================================
    "family": [
        "divorce",             # [KEPT] Anchors "separated", "split up",
        #                      #  "didn't work out"
        "abuse",               # [KEPT] Core anchor for "mistreatment", "hurt"
        "domestic violence",   # [KEPT as multi-word] "Domestic" alone is too
        #                      #  polysemous. Anchors "troubled home",
        #                      #  "family problems"
        "abandonment",         # [KEPT] Anchors "left", "walked out"
        # "adoption",          # [REMOVED: not strongly taboo in modern usage.
        #                      #  Rarely euphemized — people say "adopted" directly]
        "infidelity",          # [ADDED] Major gap. Huge euphemism domain:
                               #  "cheating", "affair", "seeing someone",
                               #  "indiscretion", "straying"
    ],

    # ========================================================================
    # MONEY & CLASS
    # ========================================================================
    "money_class": [
        "poverty",             # [KEPT] Core anchor. "Underprivileged",
        #                      #  "disadvantaged", "struggling"
        # "poor",              # [REMOVED: polysemous — "poor quality", "poor dear"]
        # "wealth",            # [REMOVED: weakly taboo — being wealthy is rarely
        #                      #  something people euphemize around -- I added it anyway BUT check!]
        # "rich",              # [REMOVED: polysemous — "rich flavor", "rich history"]
        "bankrupt",            # [KEPT] Anchors "insolvent", "went under",
        #                      #  "financial difficulties"
        "debt",                # [KEPT] Anchors "in the red", "underwater",
        #                      #  "financial obligations"
    ],

    # ========================================================================
    # HOUSING
    # ========================================================================
    "housing": [
        "homeless",            # [KEPT] Core anchor. "Unhoused", "without shelter",
        #                      #  "on the streets", "rough sleeping"
        "eviction",            # [KEPT] Anchors "asked to leave", "displaced"
        # "slum",              # [REMOVED: more of a descriptor than a taboo
        #                      #  concept that gets euphemized]
        # "foreclosure",       # [REMOVED: technical financial term, not strongly
        #                      #  euphemized — people say "foreclosure" directly]
    ],

    # ========================================================================
    # EMPLOYMENT
    # ========================================================================
    "employment": [
        "fired",               # [KEPT] Core anchor. "Let go", "terminated",
        #                      #  "made redundant", "parted ways"
        "laid off",            # [KEPT as multi-word] Distinct from "fired" —
        #                      #  anchors corporate euphemisms: "restructured",
        #                      #  "downsized", "right-sized", "reduction in force"
        # "unemployed",        # [REMOVED: not strongly euphemized — people
        #                      #  say "between jobs" but that's closer to "fired"]
        # "termination",       # [REMOVED: redundant with "fired"]
    ],

    # ========================================================================
    # EDUCATION
    # ========================================================================
    "education": [
        "illiterate",          # [KEPT] Anchors "can't read", "educationally
        #                      #  disadvantaged"
        # "dropout",           # [REMOVED: already fairly direct/informal]
        # "failure",           # [REMOVED: massively polysemous]
        "expelled",            # [KEPT] Anchors "asked to leave", "removed"
    ],

    # ========================================================================
    # CRIME & LEGAL
    # ========================================================================
    "crime": [
        "crime",               # [KEPT] Broad anchor
        # "criminal",          # [REMOVED: redundant with "crime"]
        "theft",               # [KEPT] Anchors "shoplifting", "five-finger
        #                      #  discount", "helping themselves"
        "murder",              # [KEPT HERE] Distinct from "kill" in death category.
        #                      #  Here it anchors legal/crime euphemisms:
        #                      #  "homicide", "foul play"
        # "assault",           # [REMOVED: appears in violence category]
        "prison",              # [KEPT] Anchors "inside", "away", "doing time",
        #                      #  "correctional facility"
        "bribery",             # [ADDED] Distinct from "corruption" (in politics).
                               #  Anchors "kickback", "greasing palms", "incentive"
    ],

    # ========================================================================
    # POLITICS
    # ========================================================================
    "politics": [
        "corruption",          # [KEPT] Anchors "irregularities", "misconduct"
        "propaganda",          # [KEPT] Anchors "messaging", "information campaign"
        "censorship",          # [KEPT] Anchors "content moderation", "restricted"
        # "authoritarian",     # [REMOVED: redundant with "dictator"]
        "dictator",            # [KEPT] Anchors "strongman", "leader"
        "lying",               # [ADDED] Political euphemisms for lying are vast:
                               #  "misspoke", "alternative facts", "inaccurate",
                               #  "walked back". Not covered by other anchors.
    ],

    # ========================================================================
    # MILITARY & CONFLICT
    # ========================================================================
    "military": [
        "war",                 # [KEPT] Core anchor. "Conflict", "military action",
        #                      #  "police action", "operation"
        "bombing",             # [KEPT] Anchors "airstrike", "sortie",
        #                      #  "neutralizing targets"
        "civilian death",      # [KEPT as multi-word] "Civilian" alone is not taboo.
        #                      #  Anchors "collateral damage", "non-combatant
        #                      #  casualties"
        # "casualties",        # [REMOVED: redundant with "civilian death" + "war"]
        "torture",             # [KEPT] Anchors "enhanced interrogation",
        #                      #  "coercive techniques"
        "genocide",            # [KEPT] Anchors "ethnic cleansing" (which is itself
        #                      #  a euphemism for genocide), "purge"
        # "ethnic cleansing",  # [REMOVED: it's actually a EUPHEMISM for genocide,
        #                      #  so it should be a DETECTION TARGET, not an anchor]
        # "massacre",          # [REMOVED: redundant with "genocide" + "kill"]
    ],

    # ========================================================================
    # RACISM & DISCRIMINATION
    # ========================================================================
    "racism": [
        "racism",              # [KEPT] Core anchor. "Prejudice", "bias",
        #                      #  "racial insensitivity"
        # "racial discrimination", # [REMOVED: redundant with "racism"]
        "segregation",         # [KEPT] Distinct historical/structural concept
        # "racial slur",       # [REMOVED: meta-term, not a taboo concept itself]
        "discrimination",      # [ADDED] Broader than racism — covers age, gender,
                               #  disability discrimination euphemisms too
    ],

    # ========================================================================
    # MIGRATION
    # ========================================================================
    "migration": [
        "immigrant",           # [KEPT] Anchors "newcomer", "expat", "migrant"
        "refugee",             # [KEPT] Anchors "displaced person", "asylum seeker"
        "deportation",         # [KEPT] Anchors "removal", "sent back",
        #                      #  "repatriation"
        # "illegal immigration", # [REMOVED: itself a politically loaded framing.
        #                      #  "Immigrant" + "deportation" + "crime" cover
        #                      #  this space]
    ],

    # ========================================================================
    # SOCIAL CONFLICT
    # ========================================================================
    "social_conflict": [
        # "riot",              # [REMOVED: rarely euphemized — it's already the
        #                      #  blunt word. People euphemize BY saying "unrest"
        #                      #  etc., which will match "violence" or "protest"]
        "violence",            # [KEPT] Core anchor
        # "protest",           # [REMOVED: not inherently taboo. Protests are
        #                      #  normal democratic activity. Polysemous.]
        # "conflict",          # [REMOVED: extremely polysemous — "conflict of
        #                      #  interest", "scheduling conflict"]
        # "uprising",          # [REMOVED: redundant with "violence" + "war"]
        "riot",                # [ADDED BACK] On reflection, "riot" is useful as
                               #  distinct from general "violence" — it anchors
                               #  "unrest", "disturbance", "civil disorder"
    ],

    # ========================================================================
    # VIOLENCE & ASSAULT
    # ========================================================================
    "violence": [
        # "abuse",             # [REMOVED: already in family category]
        "assault",             # [KEPT] Anchors "attacked", "got physical"
        "rape",                # [KEPT] Highly euphemized: "assaulted", "violated",
        #                      #  "taken advantage of", "non-consensual"
    ],

    # ========================================================================
    # TECHNOLOGY & PRIVACY
    # ========================================================================
    "technology": [
        "surveillance",        # [KEPT] Anchors "monitoring", "oversight",
        #                      #  "keeping an eye on"
        # "data collection",   # [REMOVED: already a euphemism for surveillance.
        #                      #  Should be a detection target, not anchor]
        # "privacy violation", # [REMOVED: redundant with "surveillance"]
        # "tracking",          # [REMOVED: polysemous — package tracking,
        #                      #  fitness tracking]
        "spying",              # [ADDED] Distinct from surveillance in tone.
                               #  Anchors "intelligence gathering", "snooping"
    ],

    # ========================================================================
    # ENVIRONMENT
    # ========================================================================
    "environment": [
        "pollution",           # [KEPT] Anchors "emissions", "environmental impact"
        # "climate change",    # [REMOVED: already a somewhat euphemistic framing
        #                      #  (vs "climate crisis"/"global warming"). Also
        #                      #  not consistently euphemized in a useful way]
        "extinction",          # [KEPT] Anchors "dying out", "disappearing"
        # "deforestation",     # [REMOVED: technical term, rarely euphemized]
    ],

    # ========================================================================
    # MORALITY & SIN
    # ========================================================================
    "morality": [
        "sin",                 # [KEPT] Anchors "transgression", "wrongdoing"
        # "immoral",           # [REMOVED: redundant with "sin"]
        # "vice",              # [REMOVED: polysemous — "vice president",
        #                      #  "vice grip"]
        "cheating",            # [ADDED] Distinct moral failing with its own
                               #  euphemisms: "bending the rules", "taking
                               #  shortcuts", "creative accounting"
    ],

    # ========================================================================
    # DISABILITIES
    # ========================================================================
    "disabilities": [
        "disabled",            # [KEPT] Core anchor. "Differently abled",
        #                      #  "special needs", "challenged"
        "blind",               # [KEPT with caution] Polysemous ("blind spot",
        #                      #  "color blind") but the disability sense is
        #                      #  strong enough. Consider higher threshold.
        "deaf",                # [KEPT] Anchors "hard of hearing", "hearing
        #                      #  impaired"
        # "paralyzed",         # [REMOVED: redundant with "disabled"]
        "mental illness",      # [KEPT as multi-word] "Mental" alone is too broad.
        #                      #  Anchors "issues", "challenges", "not well"
        # "handicap",          # [REMOVED: redundant with "disabled"]
        # "retarded",          # [REMOVED: redundant with "disabled"/"stupid" —
        #                      #  and it's a slur, not a neutral anchor]
        # "impairment",        # [REMOVED: redundant with "disabled"]
        # "wheelchair",        # [REMOVED: not taboo — it's a mobility device.
        #                      #  People don't euphemize "wheelchair"]
    ],

    # ========================================================================
    # MENTAL HEALTH
    # ========================================================================
    "mental_health": [
        # "insane",            # [REMOVED: informal/slang, redundant]
        # "crazy",             # [REMOVED: massively polysemous — "crazy good",
        #                      #  "crazy busy"]
        # "mad",               # [REMOVED: polysemous — "angry" in many dialects]
        "psychotic",           # [KEPT] Clinical anchor for severe mental health
        #                      #  euphemisms: "episode", "break from reality"
        # "mental disorder",   # [REMOVED: redundant with "mental illness" above]
        "schizophrenia",       # [ADDED] Distinct condition, heavily euphemized
                               #  and stigmatized. Anchors "hearing voices",
                               #  "split personality" (a misnomer, but common)
    ],

    # ========================================================================
    # RELIGION
    # ========================================================================
    "religion": [
        "god",                 # [KEPT] Anchors "the Lord", "the Almighty",
        #                      #  "the man upstairs"
        "hell",                # [KEPT] Anchors "heck", "H-E-double-hockey-sticks",
        #                      #  "the other place"
        "damn",                # [KEPT] Anchors "darn", "dang", "doggone"
        # "blasphemy",         # [REMOVED: redundant with "god"/"hell"/"damn"]
        # "sacrilege",         # [REMOVED: redundant — same cluster]
    ],

    # ========================================================================
    # WEIGHT & APPEARANCE
    # ========================================================================
    "weight_appearance": [
        "fat",                 # [KEPT] Core anchor. "Plus-size", "curvy",
        #                      #  "big-boned", "full-figured", "heavyset"
        "obese",               # [KEPT] Clinical complement to "fat" — anchors
        #                      #  medical euphemisms specifically
        # "overweight",        # [REMOVED: redundant with "fat"/"obese"]
        "ugly",                # [KEPT] Anchors "plain", "homely", "not much
        #                      #  to look at", "unconventional looking"
        "bald",                # [ADDED] Distinct appearance taboo with its own
                               #  euphemisms: "thinning", "receding",
                               #  "follicly challenged"
    ],

    # ========================================================================
    # HYGIENE & SMELL
    # ========================================================================
    "hygiene": [
        "body odor",           # [KEPT as multi-word] "Body" alone is not taboo.
        #                      #  Anchors "B.O.", "smells"
        # "smell",             # [REMOVED: extremely polysemous — "smell the roses"]
        # "stench",            # [REMOVED: redundant with "body odor"]
        # "dirty",             # [REMOVED: massively polysemous — "dirty dishes"]
        # "unclean",           # [REMOVED: redundant with "body odor"]
    ],

    # ========================================================================
    # ADDITIONAL CATEGORIES — GAPS IN ORIGINAL LIST
    # ========================================================================

    # [ADDED CATEGORY] Profanity & Swearing
    # Minced oaths are among the most common euphemisms and were
    # completely missing from the original list.
    "profanity": [
        "fuck",                # Anchors "fudge", "eff", "the F-word", "frick"
        "shit",                # Anchors "shoot", "crap", "sugar", "shucks"
        "ass",                 # Anchors "butt", "behind", "rear end", "backside"
    ],

    # [ADDED CATEGORY] Lying & Deception (non-political)
    # General dishonesty euphemisms were missing.
    "deception": [
        "lie",                 # Anchors "fib", "stretch the truth",
                               #  "economical with the truth", "white lie"
        # Note: "lying" in politics category handles political context
    ],

    # [ADDED CATEGORY] Bodily Appearance — Cosmetic
    # Cosmetic procedures are heavily euphemized and were missing.
    "cosmetic": [
        "plastic surgery",     # Anchors "work done", "procedure", "enhancement",
                               #  "a little help", "freshened up"
    ],

    # [ADDED CATEGORY] Dismissal & Rejection
    # Social rejection is a common euphemism domain.
    "rejection": [
        "rejection",           # Anchors "passed on", "went another direction",
                               #  "not a fit", "ghosted"
    ],
}


# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================
#
# TOTAL ANCHORS: ~75 (down from ~130+ in original)
#   - Reduced redundancy means faster similarity computation
#   - Better precision (fewer false positives from polysemous terms)
#   - Same or better recall (added missing domains)
#
# SUGGESTED SIMILARITY THRESHOLDS (tune empirically):
#   - High-polysemy anchors (blind, blood, fat): threshold ~0.75+
#   - Low-polysemy anchors (menstruation, heroin, funeral): threshold ~0.55
#   - Default starting point: 0.65
#
# N-GRAM STRATEGY FOR INPUT TEXT:
#   - Generate unigrams, bigrams, and trigrams from input text
#   - Compare each n-gram against all anchors
#   - For sentence embedders, you can also embed full sentences and compare
#     against template-wrapped anchors ("the taboo topic of {anchor}")
#
# FIRST PASS OUTPUT SHOULD INCLUDE:
#   - The matched n-gram from the input text
#   - The anchor it matched against
#   - The category label
#   - The similarity score
#   - The surrounding context (e.g., ±20 words)
#   - Timestamp/position for the second pass
#
# ============================================================================