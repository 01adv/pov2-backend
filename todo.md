User Query
↓
LLM → semantic query + filters
↓
OpenAI Embedding → vector
↓
vector db search (embedding + metadata filter)
↓
Products
↓
LLM can rerank / explain / recommend >> currently handle inside agent executor
↓
UI Display

## follow ups based on category [done]

## recommend either after one follow up question or even after first query if enough info. of product

## filtering for multiple categories if query asked [change-prompt]

## if no category mentioned user said anything >> after one follow up >> sensibly come up with recommendation

## behaviour >> to get cross category product

## when no category mentioned >> based on vibe of or whole info >> sensibly come with recommendation across categories with top_k upto 15

## for constraints >> prepare a json

combine constraints if user gives answer to followup ques and get embeddings based on it >>
eg.
user: cute brunch outfit
assistant: follow up ques
user: anything sleeveless (it's follow up ans)
assitant: combine the query >> sleeveless brunch outfits

## Direct user query

Let's say if user directly ask for a particular dress or color.

It's not working with combined filters and query.

> > > filters were not elligible as dress under 100 does not exist.

## Features to make it more robust

- Use tools one for embeddeding (already using) and another for word to word match using bm250 like stuff.
- Use agent calling to decide what to use, can use both and decide what to return based on user query.

## we can also add additional stuff to user query while getting embeddings >> for more relevancy

## create knowledge graph of the data from csv, to use along vector db, for getting comparison results and other useful query

## current observation:

cross category searches,
vague query handling
combining follow up queries
