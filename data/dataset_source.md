# SBA 928 — Public Dataset Documentation

## 1. Dataset Name

**Online Retail**

## 2. Official Source

- **Repository:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Dataset DOI:** https://doi.org/10.24432/C5BW33
- **Original creator:** Daqing Chen
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## 3. Dataset Description

The UCI Online Retail dataset contains transaction records from a UK-based, non-store online retailer. The business primarily sells gifts and other products, and some customers are wholesalers.

The dataset contains 541,909 transaction records covering December 2010 through December 2011. Its fields include invoice numbers, product codes, product descriptions, quantities, invoice dates, unit prices, customer identifiers, and customer countries.

## 4. Purpose of This Project

This dataset will support the SBA 928 assessment, Enhancing Market Research with AI Prompt Engineering. It will provide a real-world business context for developing and evaluating prompts that generate market research insights.

The project will explore:

- Consumer purchasing patterns and customer segmentation.
- Product demand and sales trends.
- Geographic differences in purchasing activity.
- Business and competitor research questions.
- Evidence-based recommendations for business strategy.

The transaction data describes historical purchasing activity. It does not directly measure customer opinions, motivations, or competitor performance.

## 5. Planned Use in Prompt Engineering

The dataset will be used to develop diverse prompts and prompt variations for different market research scenarios. Prompts will instruct the model to distinguish observed data from assumptions, explain its reasoning, and avoid unsupported conclusions.

Selected prompts and reference answers will be prepared for a controlled comparison between the original language model and the fine-tuned model.

## 6. Data Quality and Fairness Considerations

Before analysis, the project will check for missing customer identifiers, cancelled invoices, unusual quantities or prices, duplicate records, and differences in the number of transactions across countries.

The analysis will avoid treating transaction volume as a direct measure of population preferences. Geographic comparisons will account for differences in customer counts and data coverage. Conclusions will be limited to what the dataset can reasonably support.

Because the records cover a historical period and a single retailer, the findings may not represent current retail markets or all consumer groups.

## 7. Attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

The dataset is licensed under CC BY 4.0. Attribution will be retained when the dataset is used or adapted.

## 8. Project Status

The dataset has been selected and its official source documented. Downloading, data validation, preprocessing, and prompt-dataset preparation will be completed in subsequent project steps.