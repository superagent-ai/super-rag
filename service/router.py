from typing import List

from decouple import config
from semantic_router.encoders import CohereEncoder
from semantic_router.layer import RouteLayer
from semantic_router.route import Route

from models.query import RequestPayload
from service.embedding import get_encoder
from service.vector_database import VectorService, get_vector_service


def create_route_layer() -> RouteLayer:
    routes = [
        Route(
            name="summarize",
            utterances=[
                "Summmarize the following",
                "Could you summarize the",
                "Summarize",
                "Provide a summary of",
            ],
            score_threshold=0.5,
        )
    ]
    print(config("COHERE_API_KEY"))
    encoder = CohereEncoder(cohere_api_key=config("COHERE_API_KEY"))
    return RouteLayer(encoder=encoder, routes=routes)


async def get_documents(
    *, vector_service: VectorService, payload: RequestPayload
) -> List:
    chunks = await vector_service.query(input=payload.input, top_k=4)
    documents = await vector_service.convert_to_rerank_format(chunks=chunks)

    if len(documents):
        documents = await vector_service.rerank(
            query=payload.input, documents=documents
        )
    return documents


async def query(payload: RequestPayload) -> List:
    rl = create_route_layer()
    decision = rl(payload.input).name
    encoder = get_encoder(encoder_type=payload.encoder)

    if decision == "summarize":
        vector_service: VectorService = get_vector_service(
            index_name=f"{payload.index_name}summary",
            credentials=payload.vector_database,
            encoder=encoder,
        )
        return await get_documents(vector_service=vector_service, payload=payload)

    vector_service: VectorService = get_vector_service(
        index_name=payload.index_name,
        credentials=payload.vector_database,
        encoder=encoder,
    )
    return await get_documents(vector_service=vector_service, payload=payload)
