import logging
import json
from typing import Any, Dict, List
import pandas as pd

from tugraph_api import TuGraph
from models import Entity, Triplet, TemporalEvent, StatementType, Predicate
from utils import safe_iso, safe_str
from predicates import PREDICATE_GROUPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _format_value(value: Any) -> str:
    """
    一个安全的辅助函数，用于将Python值格式化为Cypher查询语句中的字符串字面量。
    """
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return "[]"

    return str(value)

def make_connection(graph_name: str = "default", memory: bool = False, refresh: bool = False) -> TuGraph:
    """
    创建并返回一个TuGraph连接实例，如果refresh=True，则重建Schema。
    """
    conn = TuGraph(graph_name=graph_name)
    
    if refresh:
        logging.warning(f"正在彻底清空图谱 '{graph_name}'...")
        conn.call_cypher("MATCH (n) DETACH DELETE n")
        logging.info(f"图谱 '{graph_name}' 的数据已清空。")

        try:
            conn.call_cypher("CALL db.deleteLabel('vertex', 'Transcript')")
            conn.call_cypher("CALL db.deleteLabel('vertex', 'Chunk')")
            conn.call_cypher("CALL db.deleteLabel('vertex', 'Event')")
            conn.call_cypher("CALL db.deleteLabel('vertex', 'Entity')")
            conn.call_cypher("CALL db.deleteLabel('edge', 'HAS_CHUNK')")
            conn.call_cypher("CALL db.deleteLabel('edge', 'CONTAINS_EVENT')")
            for p in Predicate:
                 conn.call_cypher(f"CALL db.deleteLabel('edge', '{p.value}')")
        except Exception as e:
            logging.info(f"清理旧标签时发生可忽略的错误: {e}")

        logging.info("正在创建新的图模式 (Schema)...")
        
        # 创建顶点标签
        conn.create_vertex_label('Transcript', 'id', [('id', 'STRING', False), ('text', 'STRING', True), ('company', 'STRING', True), ('date', 'STRING', True), ('quarter', 'STRING', True)])
        conn.create_vertex_label('Chunk', 'id', [('id', 'STRING', False), ('transcript_id', 'STRING', True), ('text', 'STRING', True), ('metadata', 'STRING', True)])
        conn.create_vertex_label('Event', 'id', [('id', 'STRING', False), ('chunk_id', 'STRING', True), ('statement', 'STRING', True), ('embedding', 'STRING', True), ('triplets', 'STRING', True), ('statement_type', 'STRING', True), ('temporal_type', 'STRING', True), ('created_at', 'STRING', True), ('valid_at', 'STRING', True), ('expired_at', 'STRING', True), ('invalid_at', 'STRING', True), ('invalidated_by', 'STRING', True)])
        conn.create_vertex_label('Entity', 'id', [
            ('id', 'STRING', False), 
            ('name', 'STRING', True), 
            ('resolved_id', 'STRING', True),
            ('is_canonical', 'BOOL', True)
        ])
        
        # 创建边标签
        conn.create_edge_label('HAS_CHUNK', '[["Transcript", "Chunk"]]')
        conn.create_edge_label('CONTAINS_EVENT', '[["Chunk", "Event"]]')
        for predicate in Predicate:
            conn.create_edge_label(predicate.value, '[["Entity", "Entity"]]', ('id', 'STRING', True), ('event_id', 'STRING', True), ('value', 'STRING', True))

        logging.info("Schema 创建完成。")
        
    return conn

def insert_transcript(conn: TuGraph, transcript_data: Dict[str, Any]) -> None:
    """将 Transcript 数据作为节点插入图中。"""
    conn.insert_node(0, 'Transcript', **transcript_data)

def insert_chunk(conn: TuGraph, chunk_data: Dict[str, Any]) -> None:
    """将 Chunk 数据作为节点插入图中，并与它的 Transcript 建立关联。"""
    conn.insert_node(0, 'Chunk', **chunk_data)
    conn.insert_edge(0,
                    start_label='Transcript', start_properties={'id': chunk_data["transcript_id"]},
                    end_label='Chunk', end_properties={'id': chunk_data["id"]},
                    edge_label='HAS_CHUNK')

def insert_event(conn: TuGraph, event_data: Dict[str, Any]) -> None:
    """将 Event 数据作为节点插入图中，并与它的 Chunk 建立关联。"""
    conn.insert_node(0, 'Event', **event_data)
    conn.insert_edge(0,
                    start_label='Chunk', start_properties={'id': event_data["chunk_id"]},
                    end_label='Event', end_properties={'id': event_data["id"]},
                    edge_label='CONTAINS_EVENT')

def insert_entity(conn: TuGraph, entity_data: Dict[str, Any]) -> None:
    """使用 MERGE 操作插入或更新实体节点，以避免重复。"""
    entity_id_str = json.dumps(str(entity_data.get('id')))
    name_str = json.dumps(str(entity_data.get('name')))
    resolved_id_str = json.dumps(str(entity_data.get('resolved_id')))
    
    cypher_query = f"""
    MERGE (e:Entity {{id: {entity_id_str}}})
    ON CREATE SET e.name = {name_str}, e.resolved_id = {resolved_id_str}
    ON MATCH SET e.name = {name_str}, e.resolved_id = {resolved_id_str}
    """
    conn.call_cypher(cypher_query)

def insert_triplet(conn: TuGraph, triplet_data: Dict[str, Any]) -> None:
    """将三元组（Subject-Predicate-Object）作为边插入图中。"""
    predicate = triplet_data['predicate']
    if isinstance(predicate, Predicate):
        predicate = predicate.value

    if not isinstance(predicate, str):
        logging.error(f"无效的 predicate: {predicate}。跳过三元组插入。")
        return

    edge_props = {
        "id": safe_str(triplet_data.get("id")),
        "event_id": safe_str(triplet_data.get("event_id")),
        "value": triplet_data.get("value")
    }
    edge_props = {k: v for k, v in edge_props.items() if v is not None}
    
    conn.insert_edge(0,
                    start_label='Entity', start_properties={'id': triplet_data["subject_id"]},
                    end_label='Entity', end_properties={'id': triplet_data["object_id"]},
                    edge_label=predicate,
                    **edge_props)

def get_all_canonical_entities(conn: TuGraph) -> List[Entity]:
    """获取所有规范化的实体 (最终版：查询 is_canonical 属性)"""
    cypher_query = "MATCH (e:Entity) WHERE e.is_canonical = true RETURN e.id as id, e.name as name, e.resolved_id as resolved_id, e.type as type"
    result = conn.call_cypher(cypher_query)
    return [Entity(**data) for data in result] if result else []

def insert_canonical_entity(conn: TuGraph, entity_data: Dict[str, Any]) -> None:
    """将一个实体标记为“规范化” (最终版：设置 is_canonical 属性)"""
    entity_id_str = json.dumps(str(entity_data.get('id')))
    name_str = json.dumps(str(entity_data.get('name')))

    cypher_query = f"""
    MATCH (e:Entity {{id: {entity_id_str}}})
    SET e.is_canonical = true, e.name = {name_str}
    """
    conn.call_cypher(cypher_query)

def update_entity_references(conn: TuGraph, old_id: str, new_id: str) -> None:
    """将一个实体的 resolved_id 更新为新的规范化 ID。"""
    old_id_str = _format_value(old_id)
    new_id_str = _format_value(new_id)
    cypher_query = f"MATCH (e:Entity {{id: {old_id_str}}}) SET e.resolved_id = {new_id_str}"
    conn.call_cypher(cypher_query)

def remove_entity(conn: TuGraph, entity_id: str) -> None:
    """从图中删除一个实体及其所有关联的边。"""
    conn.delete_node(False, 'Entity', id=entity_id)

def update_events_batch(conn: TuGraph, events_to_update: List[Dict[str, Any]]) -> None:
    """通过循环逐个更新事件的失效信息。"""
    if not events_to_update:
        return
    for event in events_to_update:
        event_id = _format_value(safe_str(event.get("id")))
        invalid_at = _format_value(safe_iso(event.get("invalid_at")))
        invalidated_by = _format_value(safe_str(event.get("invalidated_by")))
        cypher_query = f"""
        MATCH (ev:Event {{id: {event_id}}})
        SET ev.invalid_at = {invalid_at}, ev.invalidated_by = {invalidated_by}
        """
        conn.call_cypher(cypher_query)
    logging.info(f"发送了 {len(events_to_update)} 个 Event 节点的更新命令。")

def has_events(conn: TuGraph) -> bool:
    """检查图中是否存在任何 Event 节点。"""
    cypher_query = "MATCH (e:Event) RETURN count(e) > 0"
    result = conn.call_cypher(cypher_query)
    if result and isinstance(result, list) and result[0] and isinstance(result[0], list):
        return result[0][0]
    return False

def view_db_table(conn: TuGraph, table_name: str, max_rows: int = 10) -> pd.DataFrame:
    """
    Uses Cypher to query data from the graph and returns it as a DataFrame,
    simulating viewing a 'table'.
    """
    label = table_name.capitalize()
    
    cypher_query = ""
    columns = []

    if table_name == "triplets":
        columns = ["id", "event_id", "subject", "predicate", "object", "value"]
        cypher_query = f"""
            MATCH (s:Entity)-[r]->(o:Entity)
            RETURN
                r.id as id,
                r.event_id as event_id,
                s.name as subject,
                type(r) as predicate,
                o.name as object,
                r.value as value
            LIMIT {max_rows}
        """
    else:
        # For nodes, the result is a dict inside a list, which is fine
        cypher_query = f"MATCH (n:{label}) RETURN n LIMIT {max_rows}"

    try:
        result = conn.call_cypher(cypher_query)
        if not result:
            return pd.DataFrame()

        if table_name == "triplets":
            # Create the DataFrame with the explicit column names
            return pd.DataFrame(result, columns=columns)
        else:
            # For node queries, the result is a list of dicts, which works automatically
            data = [row[0] for row in result] # Assuming result is [[{'prop': 'val'}]]
            return pd.DataFrame(data)
            
    except Exception as e:
        print(f"查询 '{table_name}' 时出错: {e}")
        return pd.DataFrame()