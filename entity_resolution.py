# import string
# import json
# from tugraph_api import TuGraph

# from rapidfuzz import fuzz

# from db_interface import (
#     get_all_canonical_entities,
#     insert_canonical_entity,
#     remove_entity,
#     update_entity_references,
# )
# from models import Entity

# class EntityResolution:
#     """
#     Entity resolution class.
#     """

#     def __init__(self, conn: TuGraph):
#         self.conn = conn
#         self.global_canonicals: list[Entity] = get_all_canonical_entities(conn)
#         self.threshold = 80.0
#         self.acronym_thresh = 98.0


#     def resolve_entities_batch(
#         self, batch_entities: list[Entity],
#     ) -> None:
#         """
#         Orchestrate the scalable entity resolution workflow for a batch of entities.
#         """
#         type_groups = {t: [e for e in batch_entities if e.type == t] for t in set(e.type for e in batch_entities)}

#         for entities in type_groups.values():
#             clusters = self.group_entities_by_fuzzy_match(entities)

#             for group in clusters.values():
#                 if not group:
#                     continue
#                 local_canon = self.set_medoid_as_canonical_entity(group)
#                 if local_canon is None:
#                     continue

#                 match = self.match_to_canonical_entity(local_canon, self.global_canonicals)
#                 if " " in local_canon.name:  # Multi-word entity
#                     acronym = "".join(word[0] for word in local_canon.name.split())
#                     acronym_match = next(
#                         (c for c in self.global_canonicals if fuzz.ratio(acronym, c.name) >= self.acronym_thresh and " " not in c.name), None
#                     )
#                     if acronym_match:
#                         match = acronym_match

#                 if match:
#                     canonical_id = match.id
#                 else:
#                     insert_canonical_entity(
#                         self.conn,
#                         {
#                             "id": str(local_canon.id),
#                             "name": local_canon.name,
#                             "type": local_canon.type,
#                             "description": local_canon.description,
#                         },
#                     )
#                     canonical_id = local_canon.id
#                     self.global_canonicals.append(local_canon)

#                 for entity in group:
#                     entity.resolved_id = canonical_id
                    
#                     # 安全地将变量格式化到字符串中
#                     # 使用 json.dumps 可以正确处理引号等特殊字符
#                     entity_id_str = json.dumps(str(entity.id))
#                     resolved_id_str = json.dumps(str(canonical_id))

#                     # 构建完整的Cypher查询字符串
#                     cypher_query = f"MATCH (e:Entity {{id: {entity_id_str}}}) SET e.resolved_id = {resolved_id_str}"
                    
#                     # 现在只传递一个参数给 call_cypher
#                     self.conn.call_cypher(cypher_query)

#         # Clean up any acronym duplicates after processing all entities
#         self.merge_acronym_canonicals()


#     def group_entities_by_fuzzy_match(
#             self, entities: list[Entity],
#      ) -> dict[str, list[Entity]]:
#         """
#         Group entities by fuzzy name similarity using rapidfuzz"s partial_ratio.
#         Returns a mapping from canonical name to list of grouped entities.
#         """
#         def clean(name: str) -> str:
#             return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

#         name_to_entities: dict[str, list[Entity]] = {}
#         cleaned_name_map: dict[str, str] = {}
#         for entity in entities:
#             name_to_entities.setdefault(entity.name, []).append(entity)
#             cleaned_name_map[entity.name] = clean(entity.name)
#         unique_names = list(name_to_entities.keys())

#         clustered: dict[str, list[Entity]] = {}
#         used = set()
#         for name in unique_names:
#             if name in used:
#                 continue
#             clustered[name] = []
#             for other_name in unique_names:
#                 if other_name in used:
#                     continue
#                 score = fuzz.partial_ratio(cleaned_name_map[name], cleaned_name_map[other_name])
#                 if score >= self.threshold:
#                     clustered[name].extend(name_to_entities[other_name])
#                     used.add(other_name)
#         return clustered


#     def set_medoid_as_canonical_entity(self, entities: list[Entity]) -> Entity | None:
#         """
#         Select as canonical the entity in the group with the highest total similarity (sum of partial_ratio) to all others.
#         Returns the medoid entity or None if the group is empty.
#         """
#         if not entities:
#             return None

#         def clean(name: str) -> str:
#             return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

#         n = len(entities)
#         scores = [0.0] * n
#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     s1 = clean(entities[i].name)
#                     s2 = clean(entities[j].name)
#                     scores[i] += fuzz.partial_ratio(s1, s2)
#         max_idx = max(range(n), key=lambda idx: scores[idx])
#         return entities[max_idx]


#     def match_to_canonical_entity(self, entity: Entity, canonical_entities: list[Entity]) -> Entity | None:
#         """
#         Fuzzy match a single entity to a list of canonical entities.
#         Returns the best matching canonical entity or None if no match above self.threshold.
#         """
#         def clean(name: str) -> str:
#             return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

#         best_score: float = 0
#         best_canon = None
#         for canon in canonical_entities:
#             score = fuzz.partial_ratio(clean(entity.name), clean(canon.name))
#             if score > best_score:
#                 best_score = score
#                 best_canon = canon
#         if best_score >= self.threshold:
#             return best_canon
#         return None


#     def merge_acronym_canonicals(self) -> None:
#         """
#         Merge canonical entities where one is an acronym of another.
#         """
#         multi_word = [e for e in self.global_canonicals if " " in e.name]
#         single_word = [e for e in self.global_canonicals if " " not in e.name]

#         acronym_map = {}
#         for entity in multi_word:
#             acronym = "".join(word[0].upper() for word in entity.name.split())
#             acronym_map[entity.id] = acronym

#         for entity in multi_word:
#             acronym = acronym_map[entity.id]
#             for single_entity in single_word:
#                 score = fuzz.ratio(acronym, single_entity.name)
#                 if score >= self.threshold:
#                     update_entity_references(self.conn, str(entity.id), str(single_entity.id))
#                     remove_entity(self.conn, str(entity.id))
#                     self.global_canonicals.remove(entity)
#                     break
import string
import json
from tugraph_api import TuGraph

from rapidfuzz import fuzz

from db_interface import (
    get_all_canonical_entities,
    insert_canonical_entity,
    remove_entity,
    update_entity_references,
)
from models import Entity

class EntityResolution:
    """
    Entity resolution class.
    """

    def __init__(self, conn: TuGraph):
        self.conn = conn
        self.global_canonicals: list[Entity] = get_all_canonical_entities(conn)
        self.threshold = 80.0
        self.acronym_thresh = 98.0

    def resolve_entities_batch(
        self, batch_entities: list[Entity],
    ) -> None:
        """
        Orchestrate the scalable entity resolution workflow for a batch of entities.
        """
        type_groups = {t: [e for e in batch_entities if e.type == t] for t in set(e.type for e in batch_entities)}

        for entities in type_groups.values():
            clusters = self.group_entities_by_fuzzy_match(entities)

            for group in clusters.values():
                if not group:
                    continue
                local_canon = self.set_medoid_as_canonical_entity(group)
                if local_canon is None:
                    continue

                match = self.match_to_canonical_entity(local_canon, self.global_canonicals)
                if " " in local_canon.name:  # Multi-word entity
                    acronym = "".join(word[0] for word in local_canon.name.split())
                    acronym_match = next(
                        (c for c in self.global_canonicals if fuzz.ratio(acronym, c.name) >= self.acronym_thresh and " " not in c.name), None
                    )
                    if acronym_match:
                        match = acronym_match

                if match:
                    canonical_id = match.id
                else:
                    insert_canonical_entity(
                        self.conn,
                        {
                            "id": str(local_canon.id),
                            "name": local_canon.name,
                            "type": local_canon.type,
                            "description": local_canon.description,
                        },
                    )
                    canonical_id = local_canon.id
                    self.global_canonicals.append(local_canon)

                for entity in group:
                    entity.resolved_id = canonical_id
                    
                    entity_id_str = json.dumps(str(entity.id))
                    resolved_id_str = json.dumps(str(canonical_id))

                    cypher_query = f"MATCH (e:Entity {{id: {entity_id_str}}}) SET e.resolved_id = {resolved_id_str}"
                    self.conn.call_cypher(cypher_query)
                    
        self.merge_acronym_canonicals()


    def group_entities_by_fuzzy_match(
            self, entities: list[Entity],
     ) -> dict[str, list[Entity]]:
        """
        Group entities by fuzzy name similarity using rapidfuzz"s partial_ratio.
        Returns a mapping from canonical name to list of grouped entities.
        """
        def clean(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        name_to_entities: dict[str, list[Entity]] = {}
        cleaned_name_map: dict[str, str] = {}
        for entity in entities:
            name_to_entities.setdefault(entity.name, []).append(entity)
            cleaned_name_map[entity.name] = clean(entity.name)
        unique_names = list(name_to_entities.keys())

        clustered: dict[str, list[Entity]] = {}
        used = set()
        for name in unique_names:
            if name in used:
                continue
            clustered[name] = []
            for other_name in unique_names:
                if other_name in used:
                    continue
                score = fuzz.partial_ratio(cleaned_name_map[name], cleaned_name_map[other_name])
                if score >= self.threshold:
                    clustered[name].extend(name_to_entities[other_name])
                    used.add(other_name)
        return clustered


    def set_medoid_as_canonical_entity(self, entities: list[Entity]) -> Entity | None:
        """
        Select as canonical the entity in the group with the highest total similarity (sum of partial_ratio) to all others.
        Returns the medoid entity or None if the group is empty.
        """
        if not entities:
            return None

        def clean(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        n = len(entities)
        scores = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    s1 = clean(entities[i].name)
                    s2 = clean(entities[j].name)
                    scores[i] += fuzz.partial_ratio(s1, s2)
        max_idx = max(range(n), key=lambda idx: scores[idx])
        return entities[max_idx]


    def match_to_canonical_entity(self, entity: Entity, canonical_entities: list[Entity]) -> Entity | None:
        """
        Fuzzy match a single entity to a list of canonical entities.
        Returns the best matching canonical entity or None if no match above self.threshold.
        """
        def clean(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        best_score: float = 0
        best_canon = None
        for canon in canonical_entities:
            score = fuzz.partial_ratio(clean(entity.name), clean(canon.name))
            if score > best_score:
                best_score = score
                best_canon = canon
        if best_score >= self.threshold:
            return best_canon
        return None


    def merge_acronym_canonicals(self) -> None:
        """
        Merge canonical entities where one is an acronym of another.
        """
        multi_word = [e for e in self.global_canonicals if " " in e.name]
        single_word = [e for e in self.global_canonicals if " " not in e.name]

        acronym_map = {}
        for entity in multi_word:
            acronym = "".join(word[0].upper() for word in entity.name.split())
            acronym_map[entity.id] = acronym

        for entity in multi_word:
            acronym = acronym_map[entity.id]
            for single_entity in single_word:
                score = fuzz.ratio(acronym, single_entity.name)
                if score >= self.threshold:
                    update_entity_references(self.conn, str(entity.id), str(single_entity.id))
                    remove_entity(self.conn, str(entity.id))
                    self.global_canonicals.remove(entity)
                    break