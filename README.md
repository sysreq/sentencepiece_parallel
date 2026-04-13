# SentencePiece Parallel

**This is not an official Google product.**

## Technical highlights

- This fork utilizes multiple cores and smart buffering to divide-and-conquer large datasets. Takes a 10B token source from over an hour to under 10 minutes. Reduces memory overhead by ~50 percent.
