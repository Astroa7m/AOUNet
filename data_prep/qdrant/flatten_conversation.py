import json

input_file = r"../../data/json/tutors_modules_aou_convo_rag_data.jsonl"
output_file = r"../../data/json/tutors_modules_aou_convo_rag_data_cleaned.jsonl"

not_needed_labels = ["not_found", "not_found_steered", "correction"]
written_labels = []
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if "conversation" in data:
            label = data['intent'].strip()
            if label not in not_needed_labels:
                # just to make sure
                if label not in written_labels:
                    written_labels.append(label)
                json.dump(data["conversation"], outfile, ensure_ascii=False)
                outfile.write("\n")

print("✅ Conversations extracted to", output_file)
print(f"wrote labels {written_labels}")