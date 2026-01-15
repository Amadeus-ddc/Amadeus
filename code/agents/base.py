import logging
import os
from typing import Dict, List, Any, Union
from openai import OpenAI

logger = logging.getLogger("Amadeus.Agent")

class BaseAgent:
    def __init__(self, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        # Priority: Explicit Args > Environment Variables > Default
        base_url = api_base or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.static_prompt: str = ""
        self.reading_steiner: Dict[str, List[Dict[str, Any]]] = {} # {type: [{"id": "...", "content": "...", "help": 0, "harm": 0}]}

    def _format_steiner(self) -> str:
        # Check if there are any actual records (not just empty types)
        has_any_records = any(len(records) > 0 for records in self.reading_steiner.values())
        
        if not self.reading_steiner or not has_any_records:
            return "\n\n**READING STEINER (EVOLVED STRATEGIES):**\nNo specialized protocols established yet. Follow Standard Operating Procedures."
        
        text = "\n\n**READING STEINER (EVOLVED STRATEGIES):**\n"
        for steiner_type, records in self.reading_steiner.items():
            text += f"\n[{steiner_type}]:\n"
            for record in records:
                help_val = record.get('help', 0)
                harm_val = record.get('harm', 0)
                # [id] helpful=X harmful=Y :: content
                text += f"  [{record['id']}] helpful={help_val} harmful={harm_val} :: {record['content']}\n"
        return text

    def get_full_prompt(self) -> str:
        return self.static_prompt + self._format_steiner()

    def update_steiner(self, steiner_type: str, action: str, steiner_id: Union[str, List[str]], content: str = None, feedback: str = None):
        if steiner_type not in self.reading_steiner:
            self.reading_steiner[steiner_type] = []
        
        records = self.reading_steiner[steiner_type]
        
        if action == "ADD":
            if not any(r['id'] == steiner_id for r in records):
                records.append({"id": steiner_id, "content": content, "help": 0, "harm": 0})
                logger.info(f"📈 [Reading Steiner] ADDED to [{steiner_type}]: {steiner_id}")
        elif action == "UPDATE":
            for r in records:
                if r['id'] == steiner_id:
                    if content: r['content'] = content
                    logger.info(f"📉 [Reading Steiner] UPDATED [{steiner_type}] {steiner_id}")
                    break
        elif action == "DELETE":
            self.reading_steiner[steiner_type] = [r for r in records if r['id'] != steiner_id]
            logger.info(f"🗑️ [Reading Steiner] DELETED [{steiner_type}] {steiner_id}")
        elif action == "MERGE":
            if isinstance(steiner_id, list) and len(steiner_id) >= 2:
                # Use the first ID as the "survivor" or pick the one with the smallest numeric index
                # According to the prompt instructions: "the id after merge should be the smaller one"
                try:
                    sorted_ids = sorted(steiner_id, key=lambda x: int(x.split('-')[1]) if '-' in x else 999)
                except (ValueError, IndexError):
                    sorted_ids = sorted(steiner_id)
                
                target_id = sorted_ids[0]
                other_ids = sorted_ids[1:]
                
                target_record = next((r for r in records if r['id'] == target_id), None)
                if not target_record:
                    logger.warning(f"⚠️ [Reading Steiner] MERGE target {target_id} not found.")
                    return

                new_help = target_record.get('help', 0)
                new_harm = target_record.get('harm', 0)
                
                for oid in other_ids:
                    orecord = next((r for r in records if r['id'] == oid), None)
                    if orecord:
                        new_help += orecord.get('help', 0)
                        new_harm += orecord.get('harm', 0)
                
                target_record['content'] = content
                target_record['help'] = new_help
                target_record['harm'] = new_harm
                
                # Remove merged IDs
                self.reading_steiner[steiner_type] = [r for r in records if r['id'] not in other_ids]
                logger.info(f"🤝 [Reading Steiner] MERGED {other_ids} into {target_id}")

        elif action == "FEEDBACK" and feedback:
            for r in records:
                if r['id'] == steiner_id:
                    if feedback == "helpful":
                        r['help'] = r.get('help', 0) + 1
                    elif feedback == "harmful":
                        r['harm'] = r.get('harm', 0) + 1
                    logger.info(f"⭐ [Reading Steiner] FEEDBACK ({feedback}) for [{steiner_type}] {steiner_id}")
                    break

    def get_steiner_stats(self) -> Dict[str, Any]:
        """Generate statistics about the Reading Steiner (Aligned with ACE)."""
        stats = {
            'total_records': 0,
            'high_performing': 0,  # help > 5, harm < 2
            'problematic': 0,      # harm >= help
            'unused': 0,           # help + harm = 0
            'by_type': {}
        }
        
        for steiner_type, records in self.reading_steiner.items():
            if steiner_type not in stats['by_type']:
                stats['by_type'][steiner_type] = {'count': 0, 'help': 0, 'harm': 0}
            
            for r in records:
                stats['total_records'] += 1
                stats['by_type'][steiner_type]['count'] += 1
                
                h_p = r.get('help', 0)
                h_m = r.get('harm', 0)
                
                stats['by_type'][steiner_type]['help'] += h_p
                stats['by_type'][steiner_type]['harm'] += h_m
                
                if h_p > 5 and h_m < 2:
                    stats['high_performing'] += 1
                elif h_m >= h_p and h_m > 0:
                    stats['problematic'] += 1
                elif h_p + h_m == 0:
                    stats['unused'] += 1
                    
        return stats

    def reset_steiner(self):
        """Clears all Reading Steiner protocols for this agent."""
        logger.info(f"🧹 Resetting Reading Steiner for {self.__class__.__name__}")
        for k in self.reading_steiner:
            self.reading_steiner[k] = []
