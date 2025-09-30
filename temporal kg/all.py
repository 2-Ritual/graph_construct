from db_interface import (
    make_connection,
    has_events,
    insert_chunk,
    insert_entity,
    insert_event,
    insert_transcript,
    insert_triplet,
    update_events_batch,
    view_db_table,
)
from utils import safe_iso, load_transcripts_from_pickle
from models import Transcript, TemporalEvent, Chunker
from temporal_agent import TemporalAgent
from invalidation_agent import InvalidationAgent, batch_process_invalidation
from entity_resolution import EntityResolution
import json
import pickle
from tugraph_api import TuGraph
from IPython.display import display
from datasets import Dataset
from config import CHAT_MODEL, DASHSCOPE_API_KEY, BASE_URL, EMBED_MODEL
import os
import asyncio
import tiktoken
arrow_file_path = "/home/graph/construct/dataset/jlh-ibm___earnings_call/transcripts/1.1.0/0f4669f29e8cb784a3da60005a8d82f12dad102f/earnings_call-train.arrow"

async def ingest_transcript(
        transcript: Transcript,
        conn: TuGraph,
        temporal_agent: TemporalAgent,
        invalidation_agent: InvalidationAgent,
        entity_resolver: EntityResolution) -> None:
    insert_transcript(
        conn,
        {
            "id": str(transcript.id),
            "text": transcript.text,
            "company": transcript.company,
            "date": transcript.date,
            "quarter": transcript.quarter,
        },
    )

    transcript, all_events, all_triplets, all_entities = await temporal_agent.extract_transcript_events(transcript)
    entity_resolver.resolve_entities_batch(all_entities)
    name_to_canonical = {entity.name: entity.resolved_id for entity in all_entities if entity.resolved_id}
    
    # Update triplets with resolved entity IDs
    for triplet in all_triplets:
        if triplet.subject_name in name_to_canonical:
            triplet.subject_id = name_to_canonical[triplet.subject_name]
        if triplet.object_name in name_to_canonical:
            triplet.object_id = name_to_canonical[triplet.object_name]
    
    # Invalidation processing with properly resolved triplet IDs
    events_to_update: list[TemporalEvent] = []
    if has_events(conn):
        all_events, events_to_update = await batch_process_invalidation(conn, all_events, all_triplets, invalidation_agent)

    try:
        # 步骤 1: 更新已存在的事件
        if events_to_update:
            update_events_batch(conn, events_to_update)
            print(f"Updated {len(events_to_update)} existing events")

        # 步骤 2: 插入所有 Transcript 和 Chunk 节点
        for chunk in transcript.chunks or []:
            chunk_dict = chunk.model_dump()
            insert_chunk(
                conn,
                {
                    "id": str(chunk_dict["id"]),
                    "transcript_id": str(transcript.id),
                    "text": chunk_dict["text"],
                    "metadata": json.dumps(chunk_dict["metadata"]),
                },
            )
        
        # 步骤 3: 插入所有 Event 节点
        for event in all_events:
            event_dict = {
                "id": str(event.id),
                "chunk_id": str(event.chunk_id),
                "statement": event.statement,
                "embedding": pickle.dumps(event.embedding) if event.embedding is not None else None,
                "triplets": event.triplets_json,
                "statement_type": event.statement_type.value if hasattr(event.statement_type, "value") else event.statement_type,
                "temporal_type": event.temporal_type.value if hasattr(event.temporal_type, "value") else event.temporal_type,
                "created_at": safe_iso(event.created_at),
                "valid_at": safe_iso(event.valid_at),
                "expired_at": safe_iso(event.expired_at),
                "invalid_at": safe_iso(event.invalid_at),
                "invalidated_by": str(event.invalidated_by) if event.invalidated_by else None,
            }
            insert_event(conn, event_dict)

        # 步骤 4 : 首先插入所有实体节点
        unique_entities = {str(entity.id): entity for entity in all_entities}
        for entity in unique_entities.values():
            insert_entity(conn, {"id": str(entity.id), "name": entity.name, "resolved_id": str(entity.resolved_id)})
        
        # 步骤 5 : 然后插入所有三元组关系
        for triplet in all_triplets:
            try:
                insert_triplet(
                    conn,
                    {
                        "id": str(triplet.id),
                        "event_id": str(triplet.event_id),
                        "subject_name": triplet.subject_name,
                        "subject_id": str(triplet.subject_id),
                        "predicate": triplet.predicate,
                        "object_name": triplet.object_name,
                        "object_id": str(triplet.object_id),
                        "value": triplet.value,
                    },
                )
            except KeyError as e:
                print(f"KeyError: {triplet.subject_name} or {triplet.object_name} not found in name_to_canonical")
                print(f"Skipping triplet: Entity '{e.args[0]}' is unresolved.")
                continue
        
        print("\n✅ 所有数据已成功插入图中。")

    except Exception as e:
        print(f"❌ 在数据库操作期间发生错误: {e}")
        raise e
    
    # """
    # 处理单个Transcript对象，提取所有信息并存入图数据库
    # """
    # insert_transcript(
    #     conn,
    #     {
    #         "id": str(transcript.id),
    #         "text": transcript.text,
    #         "company": transcript.company,
    #         "date": transcript.date,
    #         "quarter": transcript.quarter,
    #     },
    # )

    # transcript, all_events, all_triplets, all_entities = await temporal_agent.extract_transcript_events(transcript)
    # entity_resolver.resolve_entities_batch(all_entities)
    # name_to_canonical = {entity.name: entity.resolved_id for entity in all_entities if entity.resolved_id}

    # # Update triplets with resolved entity IDs
    # for triplet in all_triplets:
    #     if triplet.subject_name in name_to_canonical:
    #         triplet.subject_id = name_to_canonical[triplet.subject_name]
    #     if triplet.object_name in name_to_canonical:
    #         triplet.object_id = name_to_canonical[triplet.object_name]


    # # Invalidation processing with properly resolved triplet IDs
    # events_to_update: list[TemporalEvent] = []
    # if has_events(conn):
    #     all_events, events_to_update = await batch_process_invalidation(conn, all_events, all_triplets, invalidation_agent)

    # # ALL DB operations happen in single transaction
    # with conn:
    #     # Update existing events first (they're already in DB)
    #     if events_to_update:
    #         update_events_batch(conn, events_to_update)
    #         print(f"Updated {len(events_to_update)} existing events")

    #     # Insert new data
    #     for chunk in transcript.chunks or []:
    #         chunk_dict = chunk.model_dump()
    #         insert_chunk(
    #             conn,
    #             {
    #                 "id": str(chunk_dict["id"]),
    #                 "transcript_id": str(transcript.id),
    #                 "text": chunk_dict["text"],
    #                 "metadata": json.dumps(chunk_dict["metadata"]),
    #             },
    #         )
    #     for event in all_events:
    #         event_dict = {
    #             "id": str(event.id),
    #             "chunk_id": str(event.chunk_id),
    #             "statement": event.statement,
    #             "embedding": pickle.dumps(event.embedding) if event.embedding is not None else None,
    #             "triplets": event.triplets_json,
    #             "statement_type": event.statement_type.value if hasattr(event.statement_type, "value") else event.statement_type,
    #             "temporal_type": event.temporal_type.value if hasattr(event.temporal_type, "value") else event.temporal_type,
    #             "created_at": safe_iso(event.created_at),
    #             "valid_at": safe_iso(event.valid_at),
    #             "expired_at": safe_iso(event.expired_at),
    #             "invalid_at": safe_iso(event.invalid_at),
    #             "invalidated_by": str(event.invalidated_by) if event.invalidated_by else None,
    #         }

    #         insert_event(conn, event_dict)
    #     for triplet in all_triplets:
    #         try:
    #             insert_triplet(
    #                 conn,
    #                 {
    #                     "id": str(triplet.id),
    #                     "event_id": str(triplet.event_id),
    #                     "subject_name": triplet.subject_name,
    #                     "subject_id": str(triplet.subject_id),
    #                     "predicate": triplet.predicate,
    #                     "object_name": triplet.object_name,
    #                     "object_id": str(triplet.object_id),
    #                     "value": triplet.value,
    #                 },
    #             )
    #         except KeyError as e:
    #             print(f"KeyError: {triplet.subject_name} or {triplet.object_name} not found in name_to_canonical")
    #             print(f"Skipping triplet: Entity '{e.args[0]}' is unresolved.")
    #             continue
    #     # Deduplicate entities by id before insert
    #     unique_entities = {}
    #     for entity in all_entities:
    #         unique_entities[str(entity.id)] = entity
    #     for entity in unique_entities.values():
    #         insert_entity(conn, {"id": str(entity.id), "name": entity.name, "resolved_id": str(entity.resolved_id)})

    return None

async def main():
    print("正在加载数据和初始化...")
    my_dataset = Dataset.from_file(arrow_file_path)
    raw_data = list(my_dataset)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunker = Chunker(EMBED_MODEL, DASHSCOPE_API_KEY, BASE_URL, tokenizer)
    transcripts = chunker.generate_transcripts_and_chunks(raw_data)
    # transcripts = load_transcripts_from_pickle("/home/graph/savings/chunk_savings")

    temporal_agent = TemporalAgent()
    invalidation_agent = InvalidationAgent()

    tugraph_conn = make_connection(memory=False, refresh=True)
    entity_resolver = EntityResolution(tugraph_conn)
    print("初始化完成，开始处理transcript...")

    await ingest_transcript(transcripts[0], tugraph_conn, temporal_agent, invalidation_agent, entity_resolver)
    
    print("\n处理完成，正在查询数据库进行验证...")
    cypher_query = "CALL db.labels()"
    all_labels = tugraph_conn.call_cypher(cypher_query)
    print("\n所有标签（节点和关系）:", all_labels)

    triplets_df = view_db_table(tugraph_conn, "triplets", max_rows=10)
    
    print("\n查询到的三元组示例:")
    display(triplets_df)

# 这是一个标准的Python脚本入口点
if __name__ == '__main__':
    asyncio.run(main())