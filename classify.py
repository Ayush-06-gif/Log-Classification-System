from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from processor_regex import classify_with_regex

LogRow = Tuple[str, str]


def _get_bert_classifier():
    from processor_bert import classify_with_bert

    return classify_with_bert


def _get_llm_classifier():
    from processor_llm import classify_with_llm

    return classify_with_llm


def classify(logs: Sequence[LogRow]) -> List[str]:
    return [classify_log(source, log_msg) for source, log_msg in logs]


def classify_log(source: str, log_msg: str) -> str:
    if source == "LegacyCRM":
        try:
            return _get_llm_classifier()(log_msg)
        except Exception:
            pass

    label = classify_with_regex(log_msg)
    if label:
        return label

    try:
        return _get_bert_classifier()(log_msg)
    except Exception:
        return "Unclassified"

def classify_csv(input_file: str | Path) -> str:
    import pandas as pd

    input_path = Path(input_file)
    if not input_path.exists():
        candidate = Path(__file__).resolve().parent / "resources" / str(input_file)
        if candidate.exists():
            input_path = candidate

    df = pd.read_csv(input_path)

    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Save the modified file
    output_path = Path(__file__).resolve().parent / "resources" / "output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return str(output_path)

if __name__ == '__main__':
    classify_csv("test.csv")
    # logs = [
    #     ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    #     ("BillingSystem", "User 12345 logged in."),
    #     ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
    #     ("AnalyticsEngine", "Backup completed successfully."),
    #     ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
    #     ("ModernHR", "Admin access escalation detected for user 9429"),
    #     ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
    #     ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
    #     ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
    #     ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    # ]
    # labels = classify(logs)
    #
    # for log, label in zip(logs, labels):
    #     print(log[0], "->", label)


