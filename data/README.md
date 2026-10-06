# Public dataset and prompt data

## Source dataset

- **Name:** Customer Support Ticket Dataset
- **Publisher:** Suraj520 on Kaggle
- **Source page:** <https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset>
- **Download:** KaggleHub dataset slug `suraj520/customer-support-ticket-dataset`
- **Use in this project:** customer-voice research examples and recorded ticket category, priority, channel, and satisfaction labels.
- **License:** consult the current Kaggle dataset page before use or redistribution. This repository does not include or redistribute the downloaded source CSV.

The intended CSV columns include `Ticket Subject` and `Ticket Description`; other supported research fields include `Product Purchased`, `Ticket Type`, `Ticket Priority`, `Ticket Channel`, and `Customer Satisfaction Rating`. Column matching ignores spaces and punctuation and accepts common aliases. The preparation command fails with the available header names if required ticket-text columns are missing.

## Processing and limits

The preparation code discards customer-name and email columns, never supplies age or gender to the model, and redacts email addresses and phone-number-like strings in ticket text. It removes duplicate/boilerplate paragraphs, replaces unresolved `{field}` placeholders, and explicitly tells the model that ticket text is untrusted data rather than instructions. By default, it selects a seeded random sample of up to 30 usable records from the full CSV before splitting them into train and validation sets. Derived JSONL files are generated under the ignored `artifacts/` directory. Age and gender are read only for an aggregate descriptive audit with groups under five records suppressed.

Tickets are self-selected interactions with support, not a representative sample of customers. Sampled text contained unresolved template placeholders and boilerplate; the source does not establish that the descriptions are authentic customer statements. Treat the data as a pipeline demonstration, not validated primary market research. Ticket volume is not market share or prevalence; satisfaction is an observed ticket field, not a causal or population-level measure. Generated task targets are rule-derived from ticket fields and should be described as supervised field extraction, not independently verified analyst judgments.

## Prompt collection

`prompt_variations.json` contains six independently worded prompts with specific output contracts: structured field coding, concise analyst coding, satisfaction banding, evidence-bounded ticket briefing, strict JSON extraction, and privacy-aware coding. All forbid guessing missing values and discourage unsupported population claims.
