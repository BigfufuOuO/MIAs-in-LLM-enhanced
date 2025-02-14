# Methods
1. Loss
2. Perplexity
3. Reference model has some different classes:
   - **Base**: Use pre-trained model as reference model, without fine-tuning.
   - **Candidate**: Simliar dataset as reference model. For example, use article summaries from remaining news categories present in the AG News corpus ("U.S.", "Europe", "Music Feeds", "Health", "Software and Development", "Entertainment") as well as the NewsCatcher dataset3containing article summaries for eight categories that highly over- lap with AG News ("Business", "Entertainment", "Health", "Nation", "Science", "Sports", "Technol- ogy", "World"). 
   - **Orcale**: Use different subset of data as reference model, with the same dataset.
4. Zlib.
5. Lowercase.
6. Window Slice.
7. LiRA Simple.
8. Neighbourhood.
9. Min-k
10. Min-k ++.
11. SVA_MIA.(NOT IMPLEMENTED)