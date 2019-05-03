import glob
import json


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''

    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class SubFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]


class InputFeatures(object):
    def __init__(self,
                 article_features,
                 question_features,
                 answer_features,
                 label
                 ):
        self.article_features = article_features
        self.question_features = question_features
        self.answer_features = answer_features
        self.label = label


def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path + "/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                # for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id=filename + '-' + str(i),
                            context_sentence=article,
                            start_ending=question,

                            ending_0=options[0],
                            ending_1=options[1],
                            ending_2=options[2],
                            ending_3=options[3],
                            label=truth))

    return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_article_length,
                                 max_question_length,
                                 max_answer_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):
        article_features = []
        question_features = []
        answer_features = []

        choice_features = []
        context_tokens = tokenizer.tokenize(example.context_sentence)
        context_tokens = _truncate_seq_pair(
            context_tokens, max_article_length - 2)
        choice_features.append(
            get_ids(tokenizer, context_tokens, max_article_length)
        )
        article_features.append(
            SubFeatures(
                example_id=example.race_id,
                choices_features=choice_features
            )
        )
        choice_features = []
        start_ending_tokens = tokenizer.tokenize(example.start_ending)
        start_ending_tokens = _truncate_seq_pair(
            start_ending_tokens, max_question_length - 2)
        choice_features.append(
            get_ids(tokenizer, start_ending_tokens, max_question_length)
        )
        question_features.append(
            SubFeatures(
                example_id=example.race_id,
                choices_features=choice_features
            )
        )

        choice_features = []
        for ending_index, ending in enumerate(example.endings):
            ending_tokens = tokenizer.tokenize(ending)
            ending_tokens = _truncate_seq_pair(
                ending_tokens, max_answer_length - 2)
            choice_features.append(
                get_ids(tokenizer, ending_tokens, max_answer_length)
            )
        answer_features.append(
            SubFeatures(
                example_id=example.race_id,
                choices_features=choice_features
            )
        )

        label = example.label
        features.append(
            InputFeatures(
                article_features=article_features,
                question_features=question_features,
                answer_features=answer_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens


def get_ids(tokenizer, tokens, max_seq_length):
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return (tokens, input_ids, input_mask, segment_ids)
