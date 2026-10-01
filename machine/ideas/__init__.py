"""The idea form, the compiler that turns it into blocks, and each idea's saved files (L7a, L7b)."""
# form.py     Idea - one idea of Tejas's: his words, the ORDERED steps (block, timeframe, maximum wait), entry, stop or
#             hedge, targets, do-not-enter rules, filters, the <= 8 slices, the unclear parts with their readings.
#             Idea.load() gives a readable FormError for a bad form. apply_changes / with_readings / reading_choices.
# compile.py  compile_idea(form, Market) -> Signals for the tester, from the blocks only; refuses a change to one of
#             his FIXED rules with his words, and names a missing block (MissingBlockError). describe() lists every
#             block, rule and filter a form may name.
# store.py    plan/work/ideas/idea-XXX/: form.json, state.json, answers.md (append only), questions/, report.html.
