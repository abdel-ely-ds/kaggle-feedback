from spacy import displacy

colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000',
            'Other': '#007f00',
         }

def visualize(transformed_df, text_id, text):
    ents = []
    example = transformed_df[transformed_df.id == text_id]

    for start, end, label in zip(example.iloc[0]['starts'],
                                 example.iloc[0]['ends'],
                                 example.iloc[0]['classlist']
                                 ):
        ents.append(
            {
                'start': int(start),
                'end': int(end),
                'label': label
            }
        )

    doc2 = {
        "text": text,
        "ents": ents,
        "title": text_id
    }

    options = {"ents": list(colors.keys()), "colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)