import requests
import json
import logging
import base64
from datetime import datetime

logging.basicConfig(
    filename='info',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
TUGRAH_HOST = "http://localhost:7070"
USERNAME = "admin"
PASSWORD = "73@TuGraph"

class TuGraph:
    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.jwt_token = self.login()
        if not self.jwt_token:
            print(TUGRAH_HOST)
            print(USERNAME)
            print(PASSWORD)
            raise Exception("登录失败，无法获取 Token")
        else:
            print("登陆成功")

    def login(self):
        url = f"{TUGRAH_HOST}/login"
        headers = {"Content-Type": "application/json"}
        data = {"user": USERNAME, "password": PASSWORD}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            token = result["jwt"]
            if token != "":
                logging.info("登录成功，获取到 Token")
                return result["jwt"]  # 返回 JWT 令牌
            else:
                raise Exception(f"登录失败: {result.get('errorMessage')}")
        except Exception as e:
            logging.info(f"登录请求异常: {str(e)}")
            return None

    def call_cypher(self, cypher_query):
        """
        执行一个Cypher查询 (修正版：使用 requests 的 json 参数)
        """
        url = f"{TUGRAH_HOST}/cypher"
        headers = {
            # 'Content-Type': 'application/json' 会由 json 参数自动设置
            "Authorization": f"Bearer {self.jwt_token}"
        }
        data = {
            "graph": self.graph_name, 
            "script": cypher_query
        }
        
        try:
            response = requests.post(url, headers=headers, json=data) 

            response.raise_for_status()
            result = response.json()
            
            if result.get("success") is False:
                error_message = result.get('message', '未知错误')
                raise Exception(f"TuGraph 查询失败: {error_message}")

            return result.get("result")

        except Exception as e:
            logging.error(f"Cypher 请求或处理时发生严重错误: {e}")
            logging.error(f"失败的查询: {cypher_query}")
            raise

    # 3. 用户登出（可选）
    def logout(self):
        url = f"{TUGRAH_HOST}/logout"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        
        try:
            response = requests.post(url, headers=headers)
            result = response.json()
            logging.info("登出成功" if result.get("is_admin") == True else "登出失败")
        except Exception as e:
            logging.info(f"登出请求异常: {str(e)}")

    def delete_schema(self):
        """
        删除图谱的基本结构，包括顶点标签、边标签等。
        """
        # 删除边标签
        self.delete_label('edge', '第一级')
        self.delete_label('edge', '第二级')
        self.delete_label('edge', '提炼')
        self.delete_label('edge', '分类')

        # 删除顶点标签
        self.delete_label('vertex', '问询函')
        self.delete_label('vertex', '目录')
        self.delete_label('vertex', '先例')
        self.delete_label('vertex', '关键点')
        self.delete_label('vertex', '纲要')
        
        logging.info("图谱结构删除完成")

    def delete_all_schema(self):
        """
        删除所有图谱结构，包括顶点标签和边标签。
        """
        logging.info("正在删除图谱结构...")
        self.delete_label('edge')
        self.delete_label('vertex')
        logging.info("图谱结构删除完成")

    def create_vertex_label(self, label_name, primary_field, fields):
        props = []
        for field_tuple in fields:
            props.append(f"'{field_tuple[0]}'")
            props.append(f"'{field_tuple[1]}'")
            props.append(str(field_tuple[2]).lower())
        fields_str = ", ".join(props)
        cypher_query = f"CALL db.createVertexLabel('{label_name}', '{primary_field}', {fields_str})"
        logging.info(f"正在创建顶点标签: {label_name}")
        self.call_cypher(cypher_query)
        logging.info(f"顶点标签 {label_name} 创建成功")


    def create_edge_label(self, label_name, constraints, *fields):
        if fields:
            fields_str = ", ".join([f"'{field[0]}', '{field[1]}', {str(field[2]).lower()}" for field in fields])
            cypher_query = f"CALL db.createEdgeLabel('{label_name}', '{constraints}', {fields_str})"
        else:
            cypher_query = f"CALL db.createEdgeLabel('{label_name}', '{constraints}')"
        logging.info(f"正在创建边标签: {label_name}")
        self.call_cypher(cypher_query)
        logging.info(f"边标签 {label_name} 创建成功")

    def get_edge_labels(self):
        """
        获取所有边标签
        :return: 边标签列表
        """
        cypher_query = "CALL db.edgeLabels()"
        logging.info("正在获取所有边标签...")
        result = self.call_cypher(cypher_query)
        if result is not None:
            edge_labels = [record[0] for record in result]
            logging.info(f"获取到 {len(edge_labels)} 个边标签")
            return edge_labels
        else:
            logging.info("获取边标签失败")
            return []
    
    def get_vertex_labels(self):
        """
        获取所有顶点标签
        :return: 顶点标签列表
        """
        cypher_query = "CALL db.vertexLabels()"
        logging.info("正在获取所有顶点标签...")
        result = self.call_cypher(cypher_query)
        if result is not None:
            vertex_labels = [record[0] for record in result]
            logging.info(f"获取到 {len(vertex_labels)} 个顶点标签")
            return vertex_labels
        else:
            logging.info("获取顶点标签失败")
            return []
            
    def delete_label(self, label_type, label_name):
        """
        删除指定标签或所有标签
        :param label_type: 标签类型 ('vertex' 或 'edge')
        :param label_name: 标签名称
        """
        if label_name:
            cypher_query = f"CALL db.deleteLabel('{label_type}', '{label_name}')"
            logging.info(f"正在删除 {label_type} 标签: {label_name}")
        result = self.call_cypher(cypher_query)

        if result is not None:
            logging.info(f"{label_type} 标签删除成功: {label_name}")
        else:
            logging.info(f"{label_type} 标签删除失败: {label_name}")


    def delete_all_schema(self):
        """
        删除所有图谱结构，包括顶点标签和边标签。
        """
        logging.info("正在删除图谱结构...")
        cypher_query = "CALL db.dropDB()"
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info("图谱结构删除成功")
        else:
            logging.info("图谱结构删除失败")
        logging.info("图谱结构删除完成")

    def insert_node(self, vertex_idx, label_name, **properties):
        """
        Inserts node data, now with safe handling for datetime and bytes objects.
        """
        sanitized_properties = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                sanitized_properties[key] = value.isoformat()
            # --- THIS IS THE KEY MODIFICATION ---
            elif isinstance(value, bytes):
                # Encode bytes to a Base64 string to make it JSON serializable
                sanitized_properties[key] = base64.b64encode(value).decode('utf-8')
            # --- END OF MODIFICATION ---
            else:
                sanitized_properties[key] = value

        properties_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in sanitized_properties.items()])
        
        cypher = f"CREATE (:{label_name} {{{properties_str}}})"
        cypher_query = cypher
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"{label_name} 节点插入成功")
            vertex_idx += 1
            return vertex_idx
        else:
            logging.info(f"{label_name} 节点插入失败")
            return vertex_idx

    def insert_edge(self, edge_idx, start_label, start_properties, end_label, end_properties, edge_label, **edge_properties):
        """
        插入边数据，此版本包含了对特殊数据类型（如datetime和bytes）的安全处理。

        :param start_label: 起点标签名称
        :param start_properties: 起点属性，用于在MATCH子句中定位节点
        :param end_label: 终点标签名称
        :param end_properties: 终点属性，用于在MATCH子句中定位节点
        :param edge_label: 边标签名称
        :param edge_properties: 边的属性，以关键字参数形式传入
        """
        # 格式化起点和终点节点的匹配属性
        start_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in start_properties.items()])
        end_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in end_properties.items()])
        
        # --- 数据净化层 ---
        # 在序列化为JSON之前，处理特殊的Python数据类型
        sanitized_edge_properties = {}
        for key, value in edge_properties.items():
            if isinstance(value, datetime):
                # 将 datetime 对象转换为 ISO 8601 格式的字符串
                sanitized_edge_properties[key] = value.isoformat()
            elif isinstance(value, bytes):
                # 将 bytes 对象（例如来自 pickle.dumps 的结果）进行 Base64 编码，使其变为字符串
                sanitized_edge_properties[key] = base64.b64encode(value).decode('utf-8')
            else:
                sanitized_edge_properties[key] = value

        # 格式化边的属性
        edge_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in sanitized_edge_properties.items()])
        
        # 构建最终的Cypher查询语句
        cypher_query = (
            f"MATCH (a:{start_label} {{{start_props_str}}}), (b:{end_label} {{{end_props_str}}}) "
            f"CREATE (a)-[:{edge_label} {{{edge_props_str}}}]->(b)"
        )
        
        logging.info(f"cypher_query: {cypher_query}")
        logging.info(f"正在插入边: {start_label} -> {end_label} [{edge_label}]")
        result = self.call_cypher(cypher_query)
        logging.info(f"result: {result}")
        
        if result is not None:
            logging.info(f"边 {edge_label} 插入成功: {start_properties} -> {end_properties} [{edge_properties}]")
            edge_idx += 1
            return edge_idx
        else:
            logging.info(f"边 {edge_label} 插入失败: {start_properties} -> {end_properties} [{edge_properties}]")
            return edge_idx

    def delete_relationship(self, start_label, start_properties, end_label, end_properties, relationship_label, relationship_id):
        """
        删除特定的关系
        :param start_label: 起点标签名称
        :param start_properties: 起点属性，格式为 key=value
        :param end_label: 终点标签名称
        :param end_properties: 终点属性，格式为 key=value
        :param relationship_label: 关系标签名称
        :param relationship_id: 关系的特定 ID（可选）
        """
        start_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in start_properties.items()])
        end_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in end_properties.items()])
        if start_label != "":
            start_label = ":"+ start_label
        if end_label != "":
            end_label = ":" + end_label
        if relationship_label != "":
            relationship_label = ":" + relationship_label
        
        cypher_query = (
            f"MATCH (a{start_label} {{{start_props_str}}})-[r{relationship_label}]->(b{end_label} {{{end_props_str}}}) "
        )
        cypher_query += f"WHERE r.id = {relationship_id} "
        cypher_query += "DELETE r"
        
        logging.info(f"正在删除关系: {start_label} -> {end_label} [{relationship_label}]")
        if relationship_id is not None:
            logging.info(f"关系 ID: {relationship_id}")
        result = self.call_cypher(cypher_query)
        logging.info(f"result: {result}")
        if result is not None:
            logging.info(f"关系 {relationship_label} 删除成功: {start_properties} -> {end_properties}")
        else:
            logging.info(f"关系 {relationship_label} 删除失败: {start_properties} -> {end_properties}")

    def delete_node(self, all, label_name, **properties):
        """
        删除特定的节点及其关系 (修正版：为ID查询增加引号)
        """
        if all:
            cypher_query = "MATCH (n) DETACH DELETE n"
            logging.info(f"正在删除所有节点")
            result = self.call_cypher(cypher_query)
            if result is not None:
                logging.info(f"所有节点删除成功")
            else:
                logging.info(f"所有节点删除失败")
            return

        if label_name: # 确保label_name不为空
            label_name = ":" + label_name
            
        if "id" in properties:
            # --- 核心修正：使用 json.dumps 为字符串ID值安全地添加引号 ---
            entity_id_str = json.dumps(properties['id'])
            cypher_query = f"MATCH (n{label_name}) WHERE n.id = {entity_id_str} DETACH DELETE n"
        else:
            properties_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in properties.items()])
            cypher_query = f"MATCH (n{label_name} {{{properties_str}}}) DETACH DELETE n"

        logging.info(f"正在删除节点: {label_name} 节点属性: {properties}")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"节点 {label_name} 删除成功: {properties}")
        else:
            logging.info(f"节点 {label_name} 删除失败: {properties}")

    def get_max_vertex_id(self):
        """
        获取当前图谱中最大的顶点 ID
        :return: 最大顶点 ID
        """
        cypher_query = "MATCH (n) RETURN MAX(n.id) AS max_id"
        result = self.call_cypher(cypher_query)
        logging.info(f"get_max_vertex_id result {result}")
        if result[0][0] != None:
            return result[0][0]  # 返回最大顶点 ID
        return 0  # 如果没有顶点，则返回 0

    def get_max_edge_id(self):
        """
        获取当前图谱中最大的边 ID
        :return: 最大边 ID
        """
        cypher_query = "MATCH ()-[r]->() RETURN MAX(r.id) AS max_id"
        result = self.call_cypher(cypher_query)
        if result[0][0] != None:
            return result[0][0]  # 返回最大边 ID
        return 0  # 如果没有边，则返回 0

        
    def delete_duplicate_relationships(self):
        """
        查询源节点和目标节点相同的所有关系，并删除重复关系，仅保留第一条
        :return: 字典，key 为 (源节点id, 目标节点id)，value 为保留的关系 id
        """
        cypher_query = """
        MATCH (a)-[r1]->(b), (a)-[r2]->(b)
        WHERE r1.id <> r2.id
        RETURN a.id AS source_id, b.id AS target_id, r1.id AS relationship1_id, r2.id AS relationship2_id
        """
        logging.info("正在查询源节点和目标节点相同的所有关系...")
        result = self.call_cypher(cypher_query)
        
        if result is not None:
            relationships_dict = {}
            for record in result:
                source_id = record[0]
                target_id = record[1]
                relationship1_id = record[2]
                relationship2_id = record[3]
                
                key = (source_id, target_id)
                if key not in relationships_dict:
                    relationships_dict[key] = set()
                relationships_dict[key].add(relationship1_id)
                relationships_dict[key].add(relationship2_id)
            # 保留第一条关系，删除其余重复关系
            for (source_id, target_id), relationship_ids in relationships_dict.items():
                relationship_ids = list(relationship_ids)
                if len(relationship_ids) > 1:
                    # 保留第一条关系
                    relationship_id = relationship_ids[0]
                    logging.info(f"保留关系: 源节点 {source_id}, 目标节点 {target_id}, 关系 ID: {relationship_id}")
                    # 删除其余关系
                    for relationship_id_to_delete in relationship_ids[1:]:
                        logging.info(f"删除关系: 源节点 {source_id}, 目标节点 {target_id}, 关系 ID: {relationship_id_to_delete}")
                        self.delete_relationship('', {'id': source_id}, '', {'id': target_id}, '', relationship_id=relationship_id_to_delete)
                else:
                    relationship_id = relationship_ids[0]
                    logging.info(f"保留关系: 源节点 {source_id}, 目标节点 {target_id}, 关系 ID: {relationship_id}")
                    # 如果只有一条关系，直接保留，不需要删除

            logging.info(f"查询并处理了 {len(relationships_dict)} 对重复关系")
            return relationships_dict
        else:
            logging.info("查询失败")
            return {}

    def get_all_chunks(self, label_name):
        """获取指定标签的所有节点
        :param label_name: 节点标签名称
        :return: 节点列表，格式为 {id: name}
        """
        if label_name != "":
            label_name = ":" + label_name
        cypher_query = f"MATCH (n{label_name}) RETURN n.chunk_id AS id, n.name AS name"
        logging.info(f"正在获取所有 {label_name} 节点...")
        result = self.call_cypher(cypher_query)
        nodes = dict()
        if result is not None:
            for record in result:
                nodes[record[0]] = record[1]
            logging.info(f"获取到 {len(nodes)} 个 {label_name} 节点")
        else:
            logging.info(f"获取 {label_name} 节点失败")
        return nodes
    
    def get_all_nodes(self, label_name):
        """获取指定标签的所有节点
        :param label_name: 节点标签名称 
        :return: 节点列表，格式为 {id: name}
        """
        if label_name != "":
            label_name = ":" + label_name
        cypher_query = f"MATCH (n{label_name}) RETURN n.id AS id, n.name AS name"
        logging.info(f"正在获取所有 {label_name} 节点...")
        result = self.call_cypher(cypher_query)
        nodes = dict()
        if result is not None:
            for record in result:
                nodes[record[0]] = record[1]
            logging.info(f"获取到 {len(nodes)} 个 {label_name} 节点")
        else:
            logging.info(f"获取 {label_name} 节点失败")
        return nodes
        
    def get_all_edges(self, edge_label=""):
        """获取指定标签的所有边
        :param edge_label: 边标签名称
        :return: 边列表，格式为 {id: name}
        """
        if edge_label != "":
            edge_label = ":" + edge_label
        cypher_query = f"MATCH ()-[e{edge_label}]->() RETURN e.edge_id AS id, e.name AS name"
        logging.info(f"正在获取所有 {edge_label} 边...")
        result = self.call_cypher(cypher_query)
        edges = dict()
        if result is not None:
            for record in result:
                edges[record[0]] = record[1]
            logging.info(f"获取到 {len(edges)} 条 {edge_label} 边")
            return edges
        else:
            logging.info(f"获取 {edge_label} 边失败")
            return None
    
    def get_all_node_chunk_content(self, label_name):
        """获取指定标签的所有节点的分块内容
        :param label_name: 节点标签名称
        :return: 节点分块内容列表，格式为 {id: chunk_content}
        """
        if label_name != "":
            label_name = ":" + label_name
        cypher_query = f"MATCH (n{label_name}) RETURN n.chunk_id AS id, n.chunk_content AS chunk_content"
        logging.info(f"正在获取所有 {label_name} 节点的分块内容...")
        result = self.call_cypher(cypher_query)
        chunk_contents = dict()
        if result is not None:
            for record in result:
                chunk_contents[record[0]] = record[1]
            logging.info(f"获取到 {len(chunk_contents)} 个 {label_name} 节点的分块内容")
        else:
            logging.info(f"获取 {label_name} 节点的分块内容失败")
        return chunk_contents

    def get_all_src_dst_nodes(self, edge_label):
        """获取指定标签的所有源节点和目标节点
        :param edge_label: 边标签名称
        :return: 源节点和目标节点列表，格式为 (src_name, dst_name)
        """
        if edge_label != "":
            edge_label = ":" + edge_label
        cypher_query = f"MATCH (src)-[e{edge_label}]->(dst) RETURN src.name AS src_name, dst.name AS dst_name"
        logging.info(f"正在获取所有 {edge_label} 边的源节点和目标节点...")
        result = self.call_cypher(cypher_query)
        src_dst_nodes = list()
        if result is not None:
            for record in result:
                src_dst_nodes.append((record[0], record[1]))
            logging.info(f"获取到 {len(src_dst_nodes)} 对 {edge_label} 边的源节点和目标节点")
        else:
            logging.info(f"获取 {edge_label} 边的源节点和目标节点失败")
        return src_dst_nodes

    def get_dst_node_ids_from_src_node_ids(self, src_label, src_node_id, edge_label, dst_label):
        """
        获取指定节点类型和 ID 的所有邻居节点 ID 列表
        :param src_label: 源节点类型
        :param src_node_id: 源节点 ID
        :param edge_label: 边类型
        :param dst_label: 目标节点类型
        """
        cypher_query = f"MATCH (src:{src_label})-[e:{edge_label}]->(dst:{dst_label}) WHERE src.id = {src_node_id} RETURN dst.id AS id"
        logging.info(f"正在查询 {src_label} {src_node_id} 的所有 {dst_label} 邻居...")
        result = self.call_cypher(cypher_query)
        if result is not None:
            neighbor_ids = [record[0] for record in result]
            logging.info(f"查询到 {len(neighbor_ids)} 个 {dst_label} 关联到 {src_label} {src_node_id}")
            return neighbor_ids
        else:
            logging.info(f"查询 {src_label} {src_node_id} 的邻居失败")
            return []

    def get_src_node_ids_from_dst_node_ids(self, src_label, dst_node_id, edge_label, dst_label):
        """
        获取指定节点类型和 ID 的所有源节点 ID 列表
        :param src_label: 源节点类型
        :param dst_node_id: 目标节点 ID
        :param edge_label: 边类型
        :param dst_label: 目标节点类型
        :return: 源节点 ID 列表
        """
        cypher_query = f"MATCH (src:{src_label})-[e:{edge_label}]->(dst:{dst_label}) WHERE dst.id = {dst_node_id} RETURN src.id AS id"
        logging.info(f"正在查询 {dst_label} {dst_node_id} 的所有 {src_label} 邻居...")
        result = self.call_cypher(cypher_query)
        if result is not None:
            neighbor_ids = [record[0] for record in result]
            logging.info(f"查询到 {len(neighbor_ids)} 个 {src_label} 关联到 {dst_label} {dst_node_id}")
            return neighbor_ids
        else:
            logging.info(f"查询 {dst_label} {dst_node_id} 的邻居失败")
            return []

    def get_all_dst_nodes_unrelated_to_src_node(self, dst_node_type, edge_type="", src_node_type=""):
        """
        获取所有不相关于源节点的目标节点
        :param dst_node_type: 目标节点类型
        :param edge_type: 边类型（可选，如果不指定则查询所有边类型）
        :param src_node_type: 源节点类型（可选，如果不指定则查询所有源节点类型）
        :return: 包含 id 和 name 的字典列表
        """
        dst_nodes_unrelated_to_src = []
        
        if edge_type and src_node_type:
            cypher_query = (
                f"MATCH (k:{dst_node_type}) "
                f"WHERE NOT EXISTS((:{src_node_type})-[:{edge_type}]->(k)) "
                f"RETURN k.id AS id, k.name AS name;"
            )
        elif edge_type:
            cypher_query = (
                f"MATCH (k:{dst_node_type}) "
                f"WHERE NOT EXISTS(()-[:{edge_type}]->(k)) "
                f"RETURN k.id AS id, k.name AS name;"
            )
        elif src_node_type:
            cypher_query = (
                f"MATCH (k:{dst_node_type}) "
                f"WHERE NOT EXISTS((:{src_node_type})-[]->(k)) "
                f"RETURN k.id AS id, k.name AS name;"
            )
        else:
            cypher_query = (
                f"MATCH (k:{dst_node_type}) "
                f"WHERE NOT EXISTS(()-[]->(k)) "
                f"RETURN k.id AS id, k.name AS name;"
            )
        
        result = self.call_cypher(cypher_query)
        if result is not None and len(result) > 0:
            for record in result:
                dst_nodes_unrelated_to_src.append({"id": record[0], "name": record[1]})
                #logging.info(f"查询到不相关于源节点的目标节点")
        else:
            logging.info("没有查询到不相关于源节点的目标节点")

        return dst_nodes_unrelated_to_src
    
    def get_all_dst_nodes_from_src_node_name(self, dst_node_type, src_node_name, edge_type="", src_node_type=""):
        """
        获取所有从源节点名称到目标节点的路径
        :param dst_node_type: 目标节点类型
        :param src_node_name: 源节点名称
        :param edge_type: 边类型（可选，如果不指定则查询所有边类型）
        :param src_node_type: 源节点类型（可选，如果不指定则查询所有源节点类型）
        :return: 包含 id 和 name 的字典
        """
        logging.info(f"正在查询从源节点 {src_node_name} 到目标节点的路径...")
        logging.info(f"查询条件: 目标节点类型: {dst_node_type}, 源节点名称: {src_node_name}, 边类型: {edge_type}, 源节点类型: {src_node_type}")
        dst_nodes = dict()
        if edge_type and src_node_type:
            cypher_query = (
                f"MATCH (src:{src_node_type})-[e:{edge_type}]->(dst:{dst_node_type}) "
                f"WHERE src.name = '{src_node_name}' "
                f"RETURN dst.id AS id, dst.name AS name;"
            )
        elif edge_type:
            cypher_query = (
                f"MATCH (src)-[e:{edge_type}]->(dst:{dst_node_type}) "
                f"WHERE src.name = '{src_node_name}' "
                f"RETURN dst.id AS id, dst.name AS name;"
            )
        elif src_node_type:
            cypher_query = (
                f"MATCH (src:{src_node_type})-[]->(dst:{dst_node_type}) "
                f"WHERE src.name = '{src_node_name}' "
                f"RETURN dst.id AS id, dst.name AS name;"
            )
        else:
            cypher_query = (
                f"MATCH (src)-[]->(dst:{dst_node_type}) "
                f"WHERE src.name = '{src_node_name}' "
                f"RETURN dst.id AS id, dst.name AS name;"
            )
        result = self.call_cypher(cypher_query)
        if result is not None and len(result) > 0:
            for record in result:
                dst_nodes[record[0]] = record[1]
            logging.info(f"查询到 {len(dst_nodes)} 个目标节点")
        else:
            logging.info("没有查询到目标节点")

        return dst_nodes
        
    def get_node_by_name(self, label_name, node_name):
        """
        根据节点标签和名称查询节点 ID
        :param label_name: 节点标签名称
        :param node_name: 节点名称
        :return: 节点 ID 如果存在，否则返回 None
        """
        if label_name != "":
            label_name = ":" + label_name
        if node_name != "":
            node_name = node_name.replace("'", "\\'")
        cypher_query = f"MATCH (n{label_name}) WHERE n.name = '{node_name}' RETURN n.id AS id"
        logging.info(f"正在查询节点是否存在: {label_name} node_name: {node_name}")
        result = self.call_cypher(cypher_query)
        if result is not None and len(result) > 0:
            node_id = result[0][0]
            logging.info(f"节点 '{label_name}' 存在，ID: {node_id}")
            return node_id
        else:
            logging.info(f"节点 '{label_name}' 不存在")
            return None

    def get_node_name_by_node_id(self, label_name, node_id):
        """
        根据节点标签和 ID 查询节点名称
        :param label_name: 节点标签名称
        :param node_id: 节点 ID
        :return: 节点名称 如果存在，否则返回 None
        """
        if label_name != "":
            label_name = ":" + label_name
        cypher_query = f"MATCH (n{label_name}) WHERE n.id = {node_id} RETURN n.name AS name"
        logging.info(f"正在查询节点名称: {label_name} node_id: {node_id}")
        result = self.call_cypher(cypher_query)
        if result is not None and len(result) > 0:
            node_name = result[0][0]
            logging.info(f"节点 '{label_name}' 存在，名称: {node_name}")
            return node_name
        else:
            logging.info(f"节点 '{label_name}' 不存在")
            return None

    def edge_exists(self, start_label, start_properties, end_label, end_properties, edge_label):
        """        检查指定的边是否存在
        :param start_label: 起点标签名称
        :param start_properties: 起点属性，格式为 key=value
        :param end_label: 终点标签名称
        :param end_properties: 终点属性，格式为 key=value
        :param edge_label: 边标签名称
        :return: 如果边存在，返回 True；否则返回 False
        """
        start_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in start_properties.items()])
        end_props_str = ", ".join([f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in end_properties.items()])
        if start_label != "":
            start_label = ":" + start_label
        if end_label != "":
            end_label = ":" + end_label
        if edge_label != "":
            edge_label = ":" + edge_label
        cypher_query = (
            f"MATCH (a{start_label} {{{start_props_str}}})-[r{edge_label}]->(b{end_label} {{{end_props_str}}}) "
            f"RETURN r.id"
        )
        logging.info(f"正在检查边是否存在: {start_label} -> {end_label} [{edge_label}]")
        result = self.call_cypher(cypher_query)
        if result is not None and len(result) > 0:
            logging.info(f"边 {edge_label} 存在: {start_properties} -> {end_properties}")
            return True
        else:
            logging.info(f"边 {edge_label} 不存在: {start_properties} -> {end_properties}")
            return False
    
class TuGraphDB:
    def __init__(self):
        self.graph_name = ""
        self.jwt_token = self.login()
        if not self.jwt_token:
            raise Exception("登录失败，无法获取 Token")

    def login(self):
        url = f"{TUGRAH_HOST}/login"
        headers = {"Content-Type": "application/json"}
        data = {"user": USERNAME, "password": PASSWORD}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            token = result["jwt"]
            if token != "":
                logging.info("登录成功，获取到 Token")
                return result["jwt"]  # 返回 JWT 令牌
            else:
                raise Exception(f"登录失败: {result.get('errorMessage')}")
        except Exception as e:
            logging.info(f"登录请求异常: {str(e)}")
            return None

    # 2. 执行 Cypher 查询
    def call_cypher(self, cypher_query):
        url = f"{TUGRAH_HOST}/cypher"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jwt_token}"  # 携带 Token
        }
        data = {
            "graph": self.graph_name, 
            "script": cypher_query
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            elapsed_time = result.get("elapsedTime")
            query_result = result.get("result")

            if query_result is not None:
                return query_result  # 返回查询结果
            else:
                raise Exception(f"查询失败: {result.get('errorMessage')}")
        except Exception as e:
            # 记录下详细的错误信息
            logging.error(f"Cypher 请求或处理时发生严重错误: {str(e)}")
            logging.error(f"失败的查询: {cypher_query}")
            
            raise

    def logout(self):
        url = f"{TUGRAH_HOST}/logout"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        
        try:
            response = requests.post(url, headers=headers)
            result = response.json()
            logging.info("登出成功" if result.get("is_admin") == True else "登出失败")
        except Exception as e:
            logging.info(f"登出请求异常: {str(e)}")

    def create_subgraph(self, graph_name, description, max_size_GB):
        """
        创建图谱
        :param graph_name: 图谱名称
        :param description: 图谱描述
        :param max_size_GB: 图谱最大大小（GB）
        """
        cypher_query = f"CALL dbms.graph.createGraph('{graph_name}', '{description}', {max_size_GB})"
        logging.info(f"正在创建图谱: {graph_name}")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"图谱 {graph_name} 创建成功")
            return True
        else:
            logging.info(f"图谱 {graph_name} 创建失败")
            return False
    
    def delete_subgraph(self, graph_name):
        """
        删除图谱
        :param graph_name: 图谱名称
        """
        cypher_query = f"CALL dbms.graph.deleteGraph('{graph_name}')"
        logging.info(f"正在删除图谱: {graph_name}")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"图谱 {graph_name} 删除成功")
        else:
            logging.info(f"图谱 {graph_name} 删除失败")
        
    def mod_subgraph(self, graph_name, description=None, max_size_GB=None):
        """
        修改图谱属性
        :param graph_name: 图谱名称
        :param description: 图谱描述（可选）
        :param max_size_GB: 图谱最大大小（GB）（可选）
        """
        if description is None and max_size_GB is None:
            logging.info("没有提供任何修改参数")
            return
        
        params = {}
        if description is not None:
            params['description'] = description
        if max_size_GB is not None:
            params['max_size_GB'] = max_size_GB
        
        cypher_query = f"CALL dbms.graph.modGraph('{graph_name}', {json.dumps(params)})"
        logging.info(f"正在修改图谱: {graph_name}")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"图谱 {graph_name} 修改成功")
        else:
            logging.info(f"图谱 {graph_name} 修改失败")
    
    def list_subgraphs(self):
        """
        列出所有图谱
        """
        cypher_query = "CALL dbms.graph.listGraphs()"
        logging.info("正在列出所有图谱...")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info(f"查询到 {len(result)} 个图谱")
            return result
        else:
            logging.info("查询图谱失败")
            return None
    # CALL db.backup('备份目录路径')
    def backup_db(self, backup_path):
        """
        备份图谱数据库
        :param backup_path: 备份目录路径
        """
        cypher_query = f"CALL db.backup('{backup_path}')"
        logging.info(f"正在备份图谱数据库到: {backup_path}")
        result = self.call_cypher(cypher_query)
        if result is not None:
            logging.info("图谱数据库备份成功")
            return True
        else:
            logging.info("图谱数据库备份失败")
            return False

if __name__ == '__main__':
    temp = TuGraph("default")