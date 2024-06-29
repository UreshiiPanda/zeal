# Zeal RAG Chatbot

#### An assorment of small programs featuring a RAG chatbot

<br><br>

#### Tech Stack:  Angular, Tailwind, FastAPI, C#



<a name="readme-top"></a>


<!-- RAG Gif -->
![kanji go gif](https://github.com/UreshiiPanda/KanjiGo/assets/39992411/123d62bb-341e-4c6b-b192-941c51e6917d)



<!-- ABOUT THE PROJECT -->
## About The Project
This app was built as part of a software internship for the purposes of practicing constructing demos and
presenting them to clients. With help from a couple of teammates, we built an Angular frontend, which makes
calls to a C# service (excluded from the Docker-Compose however), and also makes calls to a FastAPI service
which was initially constructed in Jupyter Lab in order to facilitate a RAG-based (retrieval augmented generation)
company-personalized chatbot. Data was included from a company's internal database, and this data was run through
a vector database named Pinecone in order to generate dense vectors via a cosine-similarity function which returns
vectors that are most-similar to a given query on that data. Our "Zeal" RAG chatbot takes in queries (among other
variables) from users, then generates an embedding of that query via OpenAI's embeddings model, then finds the vectors 
that are most-similar to the given embedding by routing it through Pinecone, and then re-routes this info through OpenAI's
chat completion model in order to deliver all of this info (with the added context from RAG) back to the user in a human-
palatable format that feels just like a usual LLM response, but with added, company-specific context. Further construction
could include any SQL or NoSQL database for including an entire company-wide data-set into the added context.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- How To Run -->
## How to Run:
<br>

0. While this app was not hosted, if you already have Docker installed on your machine, then this app can be simply run with the
   following instructions. The docker-compose.yml contains almost all of the Docker info needed
   to understand the context in which the app runs, including: builds, ports, network, and volume
   storage. The only exception to this are the environment variables which have been placed in a
   .env file in order to protect sensitive keys like your OPENAI_API_KEY, PINECONE_API_KEY, etc.
   The step for setting up your local .env file is included below. Note that these environment
   variables can also be moved into the docker-compose.yml file and more can be read about how to do that here:
   [Docker Env Vars](https://docs.docker.com/compose/environment-variables/set-environment-variables/). The docker-compose.yml
   file will run the Angular frontend, and the FastAPI backend for the RAG-chatbot, but it will not run the C# service as
   that was not part of the RAG project but a different API.

2. Clone all project files into a root working directory.
    ```sh
        git clone https://github.com/UreshiiPanda/zeal.git
    ```
3. Store environment variables by creating ```.env``` in that same root directory.<br>
   Place your environment variables into this file. <br>
      ```
        env="place your pinecone cloud server region here"
        pinecone_api_key="place your API key here"
        openai_api_key="place your API key here"
      ```
4. From that root directory, run docker compose:
    ```sh
        docker compose up
    ```
5. To stop the app, stop docker compose from another terminal:
    ```sh
        docker compose down
    ```
