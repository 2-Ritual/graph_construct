import asyncio
import json
from typing import Any

from jinja2 import DictLoader, Environment
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from prompts import statement_extraction_prompt, date_extraction_prompt, triplet_extraction_prompt
from models import RawStatementList, Chunk, LABEL_DEFINITIONS, RawStatement, TemporalValidityRange, RawTemporalRange, TemporalType, RawExtraction, Transcript, TemporalEvent, Triplet, Entity
from predicates import PREDICATE_DEFINITIONS, Predicate
from utils import load_transcripts_from_pickle
from config import DASHSCOPE_API_KEY, BASE_URL, CHAT_MODEL, EMBED_MODEL

# class TemporalAgent:
#     """Handles temporal-based operations for extracting and processing temporal events from text."""

#     def __init__(self) -> None:
#         """Initialize the TemporalAgent with a client."""
#         self._client = AsyncOpenAI(
#             api_key=DASHSCOPE_API_KEY,
#             base_url=BASE_URL,
#             max_retries=1,
#         )
#         self._model = CHAT_MODEL

#         self._env = Environment(loader=DictLoader({
#             "statement_extraction.jinja": statement_extraction_prompt,
#             "date_extraction.jinja": date_extraction_prompt,
#             "triplet_extraction.jinja": triplet_extraction_prompt,
#         }))
#         self._env.filters["split_and_capitalize"] = self.split_and_capitalize
#     @staticmethod
#     def split_and_capitalize(value: str) -> str:
#         """Split dict key string and reformat for jinja prompt."""
#         return " ".join(value.split("_")).capitalize()

#     async def get_statement_embedding(self, statement: str) -> list[float]:
#         """Get the embedding of a statement."""
#         response = await self._client.embeddings.create(
#             model=EMBED_MODEL,
#             input=statement,
#             dimensions=256,
#         )
#         return response.data[0].embedding

#     @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
#     async def extract_statements(
#         self,
#         chunk: Chunk,
#         inputs: dict[str, Any],
#     ) -> RawStatementList:
#         """Determine initial validity date range for a statement.

#         Args:
#             chunk (Chunk): The chunk of text to analyze.
#             inputs (dict[str, Any]): Additional input parameters for extraction.

#         Returns:
#             Statement: Statement with updated temporal range.
#         """
#         inputs["chunk"] = chunk.text

#         desired_json_schema = {
#             "type": "object",
#             "properties": {
#                 "statements": RawStatementList.model_json_schema()["properties"]["statements"]
#             },
#             "required": ["statements"],
#         }

#         template = self._env.get_template("statement_extraction.jinja")
#         prompt = template.render(
#             inputs=inputs,
#             definitions=LABEL_DEFINITIONS,
#             json_schema=json.dumps(desired_json_schema, indent=2), 
#         )

#         response = await self._client.chat.completions.create(
#             model=self._model,
#             temperature=0,
#             response_format={"type": "json_object"}, 
#             messages=[{"role": "user", "content": prompt}]
#         )

#         json_string = response.choices[0].message.content
#         if not json_string:
#             return RawStatementList(statements=[])

#         parsed_dict = json.loads(json_string)
        
#         statements = RawStatementList.model_validate(parsed_dict)
#         return statements

#         # template = self._env.get_template("statement_extraction.jinja")
#         # prompt = template.render(
#         #     inputs=inputs,
#         #     definitions=LABEL_DEFINITIONS,
#         #     json_schema=RawStatementList.model_fields,
#         # )
#         # response = await self._client.responses.parse(
#         #         model=self._model,
#         #         temperature=0,
#         #         input=prompt,
#         #         text_format=RawStatementList,
#         #     )

#         # raw_statements = response.output_parsed
#         # statements = RawStatementList.model_validate(raw_statements)
#         # return statements

#     @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
#     async def extract_temporal_range(
#         self,
#         statement: RawStatement,
#         ref_dates: dict[str, Any],
#     ) -> TemporalValidityRange:
#         """Determine initial validity date range for a statement.

#         Args:
#             statement (Statement): Statement to analyze.
#             ref_dates (dict[str, Any]): Reference dates for the statement.

#         Returns:
#             Statement: Statement with updated temporal range.
#         """
#         if statement.temporal_type == TemporalType.ATEMPORAL:
#             return TemporalValidityRange(valid_at=None, invalid_at=None)

#         template = self._env.get_template("date_extraction.jinja")
#         inputs = ref_dates | statement.model_dump()

#         prompt = template.render(
#             inputs=inputs,
#             temporal_guide={statement.temporal_type.value: LABEL_DEFINITIONS["temporal_labelling"][statement.temporal_type.value]},
#             statement_guide={statement.statement_type.value: LABEL_DEFINITIONS["episode_labelling"][statement.statement_type.value]},
#             json_schema=RawTemporalRange.model_fields,
#         )

#         response = await self._client.chat.completions.create(
#             model=self._model,
#             temperature=0,
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": prompt}]
#         )

#         json_string = response.choices[0].message.content
#         raw_validity = RawTemporalRange.model_validate(json.loads(json_string)) if json_string else None
#         temp_validity = TemporalValidityRange.model_validate(raw_validity.model_dump()) if raw_validity else TemporalValidityRange()

#         if temp_validity.valid_at is None:
#             temp_validity.valid_at = inputs["publication_date"]
#         if statement.temporal_type == TemporalType.STATIC:
#             temp_validity.invalid_at = None

#         return temp_validity

#     @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
#     async def extract_triplet(
#         self,
#         statement: RawStatement,
#         max_retries: int = 3,
#     ) -> RawExtraction:
#         """Extract triplets and entities from a statement as a RawExtraction object."""
#         template = self._env.get_template("triplet_extraction.jinja")
#         prompt = template.render(
#             statement=statement.statement,
#             json_schema=RawExtraction.model_fields,
#             predicate_instructions=PREDICATE_DEFINITIONS,
#         )

#         response = await self._client.chat.completions.create(
#             model=self._model,
#             temperature=0,
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": prompt}]
#         )

#         json_string = response.choices[0].message.content
#         if not json_string:
#             return RawExtraction(triplets=[], entities=[])
            
#         parsed_dict = json.loads(json_string)
#         extraction = RawExtraction.model_validate(parsed_dict)
#         return extraction

#     async def extract_transcript_events(
#         self,
#         transcript: Transcript,
#     ) -> tuple[Transcript, list[TemporalEvent], list[Triplet], list[Entity]]:
#         """
#         For each chunk in the transcript:
#             - Extract statements
#             - For each statement, extract temporal range and Extraction in parallel
#             - Build TemporalEvent for each statement
#             - Collect all events, triplets, and entities for later DB insertion
#         Returns the transcript, all events, all triplets, and all entities.
#         """
#         if not transcript.chunks:
#             return transcript, [], [], []
#         doc_summary = {
#             "main_entity": transcript.company or None,
#             "document_type": "Earnings Call Transcript",
#             "publication_date": transcript.date,
#             "quarter": transcript.quarter,
#             "document_chunk": None,
#         }
#         all_events: list[TemporalEvent] = []
#         all_triplets: list[Triplet] = []
#         all_entities: list[Entity] = []

#         async def _process_chunk(chunk: Chunk) -> tuple[Chunk, list[TemporalEvent], list[Triplet], list[Entity]]:
#             statements_list = await self.extract_statements(chunk, doc_summary)
#             events: list[TemporalEvent] = []
#             chunk_triplets: list[Triplet] = []
#             chunk_entities: list[Entity] = []

#             CONCURRENT_REQUESTS = 5
#             semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

#             async def _process_statement_with_semaphore(statement: RawStatement):
#                 async with semaphore:
#                     return await _process_statement(statement)

#             async def _process_statement(statement: RawStatement) -> tuple[TemporalEvent, list[Triplet], list[Entity]]:
#                 temporal_range_task = self.extract_temporal_range(statement, doc_summary)
#                 extraction_task = self.extract_triplet(statement)
#                 temporal_range, raw_extraction = await asyncio.gather(temporal_range_task, extraction_task)
#                 # Create the event first to get its id
#                 embedding = await self.get_statement_embedding(statement.statement)
#                 event = TemporalEvent(
#                     chunk_id=chunk.id,
#                     statement=statement.statement,
#                     embedding=embedding,
#                     triplets=[],
#                     valid_at=temporal_range.valid_at,
#                     invalid_at=temporal_range.invalid_at,
#                     temporal_type=statement.temporal_type,
#                     statement_type=statement.statement_type,
#                 )
#                 # Map raw triplets/entities to Triplet/Entity with event_id
#                 triplets = [Triplet.from_raw(rt, event.id) for rt in raw_extraction.triplets]
#                 entities = [Entity.from_raw(re, event.id) for re in raw_extraction.entities]
#                 event.triplets = [triplet.id for triplet in triplets]
#                 return event, triplets, entities

#             if statements_list.statements:
#                 tasks = [_process_statement_with_semaphore(stmt) for stmt in statements_list.statements]
#                 results = await asyncio.gather(*tasks)

#                 for event, triplets, entities in results:
#                     events.append(event)
#                     chunk_triplets.extend(triplets)
#                     chunk_entities.extend(entities)
#             return chunk, events, chunk_triplets, chunk_entities

#         # chunk_results = await asyncio.gather(*(_process_chunk(chunk) for chunk in transcript.chunks))

#         # 使用 for 循环来逐个处理 chunk
#         chunk_results = []
#         for chunk in transcript.chunks:
#             result = await _process_chunk(chunk)
#             chunk_results.append(result)

#         transcript.chunks = [chunk for chunk, _, _, _ in chunk_results]
#         for _, events, triplets, entities in chunk_results:
#             all_events.extend(events)
#             all_triplets.extend(triplets)
#             all_entities.extend(entities)
#         return transcript, all_events, all_triplets, all_entities

class TemporalAgent:
    """处理从文本中提取和处理时序事件的操作。"""

    def __init__(self) -> None:
        """初始化TemporalAgent客户端。"""
        self._client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=BASE_URL,
            max_retries=3,  # 建议设置多次重试以增加稳定性
        )
        self._model = CHAT_MODEL

        self._env = Environment(loader=DictLoader({
            "statement_extraction.jinja": statement_extraction_prompt,
            "date_extraction.jinja": date_extraction_prompt,
            "triplet_extraction.jinja": triplet_extraction_prompt,
        }))
        self._env.filters["split_and_capitalize"] = self.split_and_capitalize
    
    @staticmethod
    def split_and_capitalize(value: str) -> str:
        """辅助Jinja模板的过滤器。"""
        return " ".join(value.split("_")).capitalize()

    async def get_statement_embedding(self, statement: str) -> list[float]:
        """获取一个声明的嵌入向量。"""
        response = await self._client.embeddings.create(
            model=EMBED_MODEL,
            input=statement,
            # dimensions=256, # 如果不确定API是否支持，建议注释掉此行
        )
        return response.data[0].embedding

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_statements(self, chunk: Chunk, inputs: dict[str, Any]) -> RawStatementList:
        """从文本块中提取声明。"""
        inputs["chunk"] = chunk.text
        desired_json_schema = {
            "type": "object",
            "properties": {"statements": RawStatementList.model_json_schema()["properties"]["statements"]},
            "required": ["statements"],
        }
        template = self._env.get_template("statement_extraction.jinja")
        prompt = template.render(
            inputs=inputs,
            definitions=LABEL_DEFINITIONS,
            json_schema=json.dumps(desired_json_schema, indent=2),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"}, 
            messages=[{"role": "user", "content": prompt}]
        )

        json_string = response.choices[0].message.content
        if not json_string:
            return RawStatementList(statements=[])

        parsed_data = json.loads(json_string)
        
        # --- 数据修正层 ---
        if isinstance(parsed_data, list):
            parsed_dict = {"statements": parsed_data}
        else:
            parsed_dict = parsed_data
        
        return RawStatementList.model_validate(parsed_dict)

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_temporal_range(self, statement: RawStatement, ref_dates: dict[str, Any]) -> TemporalValidityRange:
        """为声明确定时间范围。"""
        if statement.temporal_type == TemporalType.ATEMPORAL:
            return TemporalValidityRange(valid_at=None, invalid_at=None)

        template = self._env.get_template("date_extraction.jinja")
        inputs = ref_dates | statement.model_dump()
        prompt = template.render(
            inputs=inputs,
            temporal_guide={statement.temporal_type.value: LABEL_DEFINITIONS["temporal_labelling"][statement.temporal_type.value]},
            statement_guide={statement.statement_type.value: LABEL_DEFINITIONS["episode_labelling"][statement.statement_type.value]},
            json_schema=RawTemporalRange.model_json_schema(),
        )

        response = await self._client.chat.completions.create(
            model=self._model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}]
        )

        json_string = response.choices[0].message.content
        
        if json_string:
            parsed_data = json.loads(json_string)
            # --- 数据修正层 ---
            if isinstance(parsed_data, list) and parsed_data:
                dict_to_validate = parsed_data[0]
            else:
                dict_to_validate = parsed_data
            raw_validity = RawTemporalRange.model_validate(dict_to_validate)
        else:
            raw_validity = None
        
        temp_validity = TemporalValidityRange.model_validate(raw_validity.model_dump()) if raw_validity else TemporalValidityRange()
        if temp_validity.valid_at is None:
            temp_validity.valid_at = inputs["publication_date"]
        if statement.temporal_type == TemporalType.STATIC:
            temp_validity.invalid_at = None
        return temp_validity

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_triplet(self, statement: RawStatement) -> RawExtraction:
        """从声明中提取三元组和实体。"""
        template = self._env.get_template("triplet_extraction.jinja")
        prompt = template.render(
            statement=statement.statement,
            json_schema=RawExtraction.model_json_schema(),
            predicate_instructions=PREDICATE_DEFINITIONS,
        )

        response = await self._client.chat.completions.create(
            model=self._model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}]
        )

        json_string = response.choices[0].message.content
        if not json_string:
            return RawExtraction(triplets=[], entities=[])
            
        parsed_dict = json.loads(json_string)
        valid_predicates = {p.value for p in Predicate}
    
        raw_triplets = parsed_dict.get("triplets", [])
        
        filtered_triplets = []
        for triplet_dict in raw_triplets:
            if triplet_dict.get("predicate") in valid_predicates:
                filtered_triplets.append(triplet_dict)
            else:
                print(f"⚠️ 警告：检测到并过滤掉一个无效谓词 '{triplet_dict.get('predicate')}'")

        parsed_dict["triplets"] = filtered_triplets
        
        extraction = RawExtraction.model_validate(parsed_dict)
        return extraction

    async def extract_transcript_events(self, transcript: Transcript) -> tuple[Transcript, list, list, list]:
        """处理一篇文稿中的所有文本块并提取所有信息。"""
        if not transcript.chunks:
            return transcript, [], [], []
        doc_summary = {
            "main_entity": transcript.company or None, "document_type": "Earnings Call Transcript",
            "publication_date": transcript.date, "quarter": transcript.quarter, "document_chunk": None,
        }
        all_events, all_triplets, all_entities = [], [], []

        async def _process_chunk(chunk: Chunk) -> tuple[Chunk, list, list, list]:
            statements_list = await self.extract_statements(chunk, doc_summary)
            events, chunk_triplets, chunk_entities = [], [], []
            semaphore = asyncio.Semaphore(5)

            async def _process_statement_with_semaphore(statement: RawStatement):
                async with semaphore:
                    return await _process_statement(statement)

            async def _process_statement(statement: RawStatement) -> tuple[TemporalEvent, list, list]:
                tasks = [self.extract_temporal_range(statement, doc_summary), self.extract_triplet(statement)]
                temporal_range, raw_extraction = await asyncio.gather(*tasks)
                embedding = await self.get_statement_embedding(statement.statement)
                event = TemporalEvent(
                    chunk_id=chunk.id, statement=statement.statement, embedding=embedding, triplets=[],
                    valid_at=temporal_range.valid_at, invalid_at=temporal_range.invalid_at,
                    temporal_type=statement.temporal_type, statement_type=statement.statement_type,
                )
                triplets = [Triplet.from_raw(rt, event.id) for rt in raw_extraction.triplets]
                entities = [Entity.from_raw(re, event.id) for re in raw_extraction.entities]
                event.triplets = [t.id for t in triplets]
                return event, triplets, entities

            if statements_list.statements:
                tasks = [_process_statement_with_semaphore(stmt) for stmt in statements_list.statements]
                results = await asyncio.gather(*tasks)
                for event, triplets, entities in results:
                    events.append(event)
                    chunk_triplets.extend(triplets)
                    chunk_entities.extend(entities)
            return chunk, events, chunk_triplets, chunk_entities

        chunk_results = []
        for chunk in transcript.chunks:
            result = await _process_chunk(chunk)
            chunk_results.append(result)

        transcript.chunks = [chunk for chunk, _, _, _ in chunk_results]
        for _, events, triplets, entities in chunk_results:
            all_events.extend(events)
            all_triplets.extend(triplets)
            all_entities.extend(entities)
        return transcript, all_events, all_triplets, all_entities
    
if __name__ == '__main__':
    transcripts = load_transcripts_from_pickle()

    async def main():
        temporal_agent = TemporalAgent()
        results = await temporal_agent.extract_transcript_events(transcripts[0]) # await
        # Parse and display the results in a nice format
        transcript, events, triplets, entities = results

        print("=== TRANSCRIPT PROCESSING RESULTS ===\n")

        print(f"📄 Transcript ID: {transcript.id}")
        print(f"📊 Total Chunks: {len(transcript.chunks) if transcript.chunks is not None else 0}")
        print(f"🎯 Total Events: {len(events)}")
        print(f"🔗 Total Triplets: {len(triplets)}")
        print(f"🏷️  Total Entities: {len(entities)}")

        print("\n=== SAMPLE EVENTS ===")
        for i, event in enumerate(events[:3]):  # Show first 3 events
            print(f"\n📝 Event {i+1}:")
            print(f"   Statement: {event.statement[:100]}...")
            print(f"   Type: {event.temporal_type}")
            print(f"   Valid At: {event.valid_at}")
            print(f"   Triplets: {len(event.triplets)}")

        print("\n=== SAMPLE TRIPLETS ===")
        for i, triplet in enumerate(triplets[:5]):  # Show first 5 triplets
            print(f"\n🔗 Triplet {i+1}:")
            print(f"   Subject: {triplet.subject_name} (ID: {triplet.subject_id})")
            print(f"   Predicate: {triplet.predicate}")
            print(f"   Object: {triplet.object_name} (ID: {triplet.object_id})")
            if triplet.value:
                print(f"   Value: {triplet.value}")

        print("\n=== SAMPLE ENTITIES ===")
        for i, entity in enumerate(entities[:5]):  # Show first 5 entities
            print(f"\n🏷️  Entity {i+1}:")
            print(f"   Name: {entity.name}")
            print(f"   Type: {entity.type}")
            print(f"   Description: {entity.description}")
            if entity.resolved_id:
                print(f"   Resolved ID: {entity.resolved_id}")

    asyncio.run(main())