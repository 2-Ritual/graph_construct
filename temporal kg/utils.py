import uuid
import pickle
from datetime import datetime
from models import Transcript, parse_date_str, Entity, TemporalEvent, Triplet
from pathlib import Path
from typing import List

def save_transcript_to_pickle(transcript: Transcript, file_path: str) -> None:
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "wb") as f:
            pickle.dump(transcript, f)
        print(f"✅ Transcript 已成功保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存 Transcript 时出错: {e}")

def save_list_to_pickle(data_list: List, file_path: str) -> None:
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data_list, f)
        print(f"✅ 列表已成功保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存列表时出错: {e}")

def safe_iso(dt: datetime | None) -> str | None:
    if isinstance(dt, str):
        dt = parse_date_str(dt)

    if isinstance(dt, datetime):
        return dt.isoformat()

    return None

def safe_str(uid: uuid.UUID | None) -> str | None:
    if uid is None:
        return None
    return str(uid)
    
def load_transcripts_from_pickle(directory_path: str) -> list[Transcript]:
    loaded_transcripts = []
    dir_path = Path(directory_path).resolve()

    for pkl_file in sorted(dir_path.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                data_from_file = pickle.load(f)

            items_to_process = []
            if isinstance(data_from_file, list):
                items_to_process.extend(data_from_file)
            else:
                items_to_process.append(data_from_file)

            for item in items_to_process:
                if isinstance(item, Transcript):
                    loaded_transcripts.append(item)
                elif isinstance(item, dict):
                    loaded_transcripts.append(Transcript(**item))
                else:
                    print(f"⚠️ Warning: Skipped an item of unexpected type '{type(item).__name__}' in {pkl_file.name}")

            print(f"✅ Loaded data from {pkl_file.name}")
        except Exception as e:
            print(f"❌ Error loading {pkl_file.name}: {e}")

    return loaded_transcripts

def load_entities_from_pickle(file_path: str) -> List[Entity]:
    file = Path(file_path)
    if not file.is_file():
        print(f"⚠️ 文件不存在: {file_path}，返回空列表。")
        return []

    try:
        with open(file, "rb") as f:
            loaded_data = pickle.load(f)
        
        if isinstance(loaded_data, list):
            # 可以在这里增加更严格的类型检查，但通常pickle能保证类型
            print(f"✅ 成功从 {file_path} 加载了 {len(loaded_data)} 个 Entity。")
            return loaded_data
        else:
            print(f"❌ 错误: {file_path} 中的数据不是一个列表，而是一个 '{type(loaded_data).__name__}'。返回空列表。")
            return []
    except Exception as e:
        print(f"❌ 加载文件 {file_path} 时出错: {e}。返回空列表。")
        return []


def load_events_from_pickle(file_path: str) -> List[TemporalEvent]:
    file = Path(file_path)
    if not file.is_file():
        print(f"⚠️ 文件不存在: {file_path}，返回空列表。")
        return []

    try:
        with open(file, "rb") as f:
            loaded_data = pickle.load(f)
        
        if isinstance(loaded_data, list):
            print(f"✅ 成功从 {file_path} 加载了 {len(loaded_data)} 个 TemporalEvent。")
            return loaded_data
        else:
            print(f"❌ 错误: {file_path} 中的数据不是一个列表。返回空列表。")
            return []
    except Exception as e:
        print(f"❌ 加载文件 {file_path} 时出错: {e}。返回空列表。")
        return []


def load_triplets_from_pickle(file_path: str) -> List[Triplet]:
    file = Path(file_path)
    if not file.is_file():
        print(f"⚠️ 文件不存在: {file_path}，返回空列表。")
        return []

    try:
        with open(file, "rb") as f:
            loaded_data = pickle.load(f)
        
        if isinstance(loaded_data, list):
            print(f"✅ 成功从 {file_path} 加载了 {len(loaded_data)} 个 Triplet。")
            return loaded_data
        else:
            print(f"❌ 错误: {file_path} 中的数据不是一个列表。返回空列表。")
            return []
    except Exception as e:
        print(f"❌ 加载文件 {file_path} 时出错: {e}。返回空列表。")
        return []
