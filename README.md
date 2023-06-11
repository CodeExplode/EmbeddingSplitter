# EmbeddingSplitter

A tiny extension for the Automatic1111 web-ui, which allows splitting multi-vector text embeddings into their components.

To help with testing out the ideas presented in the recent paper Concept Decomposition for Visual Exploration and Inspiration: https://arxiv.org/abs/2305.18203

It doesn't currently check that valid vector indices are given, nor updates the list of embeddings if it changes, so the whole UI will need to be restarted to see new embeddings.
