import random
import numpy as np
import pandas as pd
import math
import numbers


def calculate_entropy(column):
    size = len(column)
    sum = 0.0
    count_dict = {}

    for el in column:  # vytvorenie slovníka na počítanie počtu jednotlivých atribútov
        if el not in count_dict:
            count_dict[el] = 1
        else:
            count_dict[el] += 1

    for uniq_el in count_dict:  # pridanie vypočítaných hodnôt každého atribútu v stĺpci k našej entropii
        n = count_dict[uniq_el]
        fraction = n / size
        sum -= fraction * math.log(fraction, 2)

    return sum


def calculate_info_gain(dataframe, datasetEntropy):
    uniqueSet = set(dataframe[dataframe.columns[0]])  # Vytvorenie jedinečného súboru atribútov
    size = len(dataframe)
    info = 0.0

    for uniq in uniqueSet:
        subset = dataframe[
            dataframe[dataframe.columns[0]] == uniq]  # vytvorenie podmnožiny s príslušnými atribútmi ako uniq
        sizeOfSubset = len(subset)
        # print(f"{sizeOfSubset}/{size} * {calculate_entropy(subset[subset.columns[1]])}")
        info += (sizeOfSubset / size) * calculate_entropy(
            subset[subset.columns[1]])  # pravdepodobnosť * entropia podmnožiny

    # print(f"info: {info}")
    return datasetEntropy - info


def get_most_informative_feature(dataset: pd.DataFrame):
    target_column = dataset[dataset.columns[-1]]
    countColumns = dataset.shape[1]
    sizeDataset = len(dataset)
    datasetEntropy = calculate_entropy(target_column)

    max_value = -1
    idx_max = 0
    bestThreshold = None

    for i in range(countColumns - 1):  # Prejdime každý stĺpec a nájdeme stĺpec s najvyššou hodnotou info_importance
        if dataset[dataset.columns[i]].dtype == 'object':  # Výpočet info_importance pre diskrétne stĺpce
            currentDF = dataset[[dataset.columns[i], dataset.columns[-1]]]

            infoGain = calculate_info_gain(currentDF, datasetEntropy)
            feature_entropy = calculate_entropy(dataset[dataset.columns[i]])
            if infoGain == 0:
                info_importance = 0
            else:
                info_importance = infoGain / feature_entropy
            # print(f"{dataset.columns[i]}: {info_importance:.3f}")

            if info_importance > max_value:  # Globálne maximum je nadradené, ak je aktuálna hodnota info_importance vyššia
                max_value = info_importance
                idx_max = i
                bestThreshold = None
        else:  # Výpočet info_importance pre číselné stĺpce
            values_set = sorted(set(dataset[dataset.columns[i]]))  # Vytvorenie súboru jedinečných hodnôt v stĺpci
            thresholds = []

            for j in range(0, len(values_set) - 1):  # Spočítajme prahové hodnoty a pridajte ich do zoznamu
                current_threshold = (values_set[j] + values_set[j + 1]) / 2
                thresholds.append(current_threshold)

            localMaxValue = 0
            localBestThreshold = thresholds[0]

            for threshold in thresholds:
                positiveSubset = dataset[
                    dataset[dataset.columns[i]] > threshold]  # rozdelíme dátový rámec na kladný a záporný
                negativeSubset = dataset[dataset[dataset.columns[i]] < threshold]

                posTarget = positiveSubset[positiveSubset.columns[-1]]
                negTarget = negativeSubset[negativeSubset.columns[-1]]

                posEntropy = calculate_entropy(posTarget)
                negEntropy = calculate_entropy(negTarget)

                posSize = len(positiveSubset)
                negSize = len(negativeSubset)

                currentEntropy = (posSize / sizeDataset) * posEntropy + (
                            negSize / sizeDataset) * negEntropy  # Výpočet entropie po separácii
                infoGain = datasetEntropy - currentEntropy

                modifyFeatureColumn = dataset[dataset.columns[i]] > threshold
                feature_entropy = calculate_entropy(modifyFeatureColumn)

                info_importance = infoGain / feature_entropy

                if info_importance > localMaxValue:
                    localMaxValue = info_importance
                    localBestThreshold = threshold

            if localMaxValue > max_value:
                max_value = localMaxValue
                idx_max = i
                bestThreshold = localBestThreshold

    return dataset.columns[idx_max], bestThreshold


def get_subset_number(dataset, feature, symbol, number):
    if symbol == '<':
        subset = dataset[dataset[feature] < number]
    elif symbol == '>':
        subset = dataset[dataset[feature] > number]
    else:
        raise TypeError("value must contain '<' or '>' as first symbol")

    subset = subset.drop(feature, axis=1)

    return subset


def built_decision_tree(dataset):
    if len(set(dataset)) == 1:
        return dataset[dataset.columns[0]].value_counts().idxmax()

    if dataset[dataset.columns[-1]].nunique() == 1:
        return dataset.iloc[0, -1]

    list_to_remove = []
    for i in range(len(set(dataset)) - 1):
        if dataset[dataset.columns[i]].nunique() == 1:
            list_to_remove.append(dataset.columns[i])

    dataset = dataset.drop(columns=list_to_remove)

    if len(set(dataset)) == 1:
        return dataset[dataset.columns[0]].value_counts().idxmax()

    best_feature, threshold = get_most_informative_feature(dataset)
    tree = {best_feature: {}}

    if dataset[best_feature].dtype == 'object':
        feature_values = set(dataset[best_feature])
        for value in feature_values:
            subset = dataset[dataset[best_feature] == value]
            subset = subset.drop(best_feature, axis=1)
            subtree = built_decision_tree(subset)
            tree[best_feature][value] = subtree

    else:
        feature_values = [f"< {threshold}", f"> {threshold}"]

        for value in feature_values:
            subset = get_subset_number(dataset, best_feature, value[0], threshold)
            subtree = built_decision_tree(subset)
            tree[best_feature][value] = subtree

        if (not isinstance(tree[best_feature][feature_values[0]], dict) and not isinstance(
                tree[best_feature][feature_values[1]], dict)
                and tree[best_feature][feature_values[0]] == tree[best_feature][feature_values[1]]):
            return tree[best_feature][feature_values[0]]
    return tree


def predict(decisionTree, current_row):
    if not isinstance(decisionTree, dict):
        return decisionTree
    else:
        root = next(iter(decisionTree))
        featureValue = current_row[root]

        if isinstance(featureValue, numbers.Number):
            keysFloat = (decisionTree[root].keys())
            keyFloat = list(keysFloat)[0]
            keyFloat = float(keyFloat[2:])
            if featureValue < keyFloat:
                featureValue = list(keysFloat)[0]
            else:
                featureValue = list(keysFloat)[1]

        if featureValue in decisionTree[root]:
            return predict(decisionTree[root][featureValue], current_row)
        else:
            return None


# ___HYPER_PARAMETERS___
src = "Diabets_large.csv"
size_test_data = 0.3
target_column = "Outcome"
non_informative_columns = []
positive_value = 1  # Hodnoty v cieľovom stĺpci, ktoré by sa mali považovať za pozitívne
negative_value = 0  # Hodnoty v cieľovom stĺpci, ktoré by sa mali považovať za negatívne


# ___PREPARE_DATA___
data = pd.read_csv(src)  # Stiahnutie datasetu
print(f"initial size of the dataset: {data.shape[0]} rows, {data.shape[1]} columns")

data.dropna(inplace=True)  # odstránenie neúplných riadkov
data = data.drop(non_informative_columns, axis=1)
print(f"dataset size after deleting incomplete rows: {data.shape[0]} rows, {data.shape[1]} columns\n")

if target_column != data.columns[-1]:  # Ak je cieľový stĺpec nie je posledný, zmeníme ho s posledným
    old_targ = data.columns[-1]
    data[old_targ], data[target_column] = data[target_column], data[old_targ]
    data = data.rename(columns={target_column: old_targ, old_targ: target_column})
    print(f"the target column \"{target_column}\" was moved to the last position\n")

value_counts = data[
    target_column].value_counts().to_string()  # Zobrazenie počtu pozitívnych a negatívnych príkladov v súbore údajov
print(f"{value_counts}\n")

# Ak existuje veľká nerovnováha v počte kladných a záporných riadkov,
# odstránime niekoľko náhodných riadkov prevažujúcej triedy, aby sme dosiahli rovnováhu.
negative_data = data[data['Outcome'] == negative_value]
random_negative_rows = negative_data.sample(n=540)  # n je potrebné nahradiť počtom riadkov, ktoré chcete vymazať
data = data.drop(random_negative_rows.index)

value_counts = data[target_column].value_counts().to_string()  # Po vyvážení skontrolujeme počet príkladov rôznych tried
print(f"{value_counts}\n")

shuffled_data = data.sample(frac=1).reset_index(drop=True)  # Zmiešanie datasetu
size_data = len(shuffled_data)

# Rozdelenie zmiešaného súboru údajov na tréningovú a testovaciu časť
IdxSplit = round(size_data * (1.0 - size_test_data))

training_data = shuffled_data.iloc[:IdxSplit]
test_data = shuffled_data.iloc[IdxSplit:]

size_training = len(training_data)
size_test = len(test_data)
print(f"train size: {size_training}, test size: {len(test_data)}")


# ___ALGORITHM___

# Hyperparametre na vytvorenie náhodného lesa
countColumns = math.ceil(math.sqrt(len(set(training_data))) + 1)  # Počet náhodných stĺpcov pre bootstrap vzorku
countRows = round(0.7 * size_training)  # odporúčaná veľkosť 60 - 80 % veľkosti tréningových údajov
countTrees = 100

print(f"Hyperparameters:\ncount of trees: {countTrees}\nsubset columns: {countColumns}\nsubset rows: {countRows}\n")
print("Creating trees...\n")
trees = []

all_columns = set(training_data)
all_columns.discard(training_data.columns[-1])

for i in range(countTrees):
    random_columns = np.random.choice(list(all_columns), countColumns, replace=False).tolist()
    random_columns.append(training_data.columns[-1])
    random_rows = random.choices(range(size_training), k=countRows)  # Náhodný výber riadkov pre podskupinu bootstrap

    current_dataset = training_data[random_columns]
    current_dataset = current_dataset.iloc[random_rows]  # Tu z podskupiny s náhodnými riadkami vyberieme
    # náhodné stĺpce spolu s cieľovým stĺpcom

    tree = built_decision_tree(current_dataset)  # pomocou funkcie vytvoríme z podskupiny bootstrap rozhodovací strom
    trees.append(tree)  # a pridáme tento strom do zoznamu stromov

print(f"{countTrees} trees was created\n")

print(f"Testing...\n")

# Počítadlá pre metriky
true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

for i in range(size_test):  # cyklus testovania
    row = test_data.iloc[i]  # Správna trieda
    answers = []

    for tree in trees:  # Prechádzame všetky stromy a zbierame ich predpovede do zoznamu
        answers.append(predict(tree, row))

    dict_answers = {}
    for answer in answers:  # Zoskupujeme predpovede stromov do slovníka
        if answer == None:
            continue
        if answer in dict_answers:
            dict_answers[answer] += 1
        else:
            dict_answers[answer] = 0
    decision = max(dict_answers,
                   key=dict_answers.get)  # Najčastejšia odpoveď v slovníku bude našou predpovedanou triedou

    # Porovnáme predpovedanú odpoveď so skutočnou a zoskupte ju do počítadiel pre metriku
    if row.iloc[-1] == positive_value:
        if row.iloc[-1] == decision:
            true_positive += 1
        else:
            false_negative += 1
    else:
        if row.iloc[-1] == decision:
            true_negative += 1
        else:
            false_positive += 1

print(f"TP: {true_positive}, FP: {false_positive}, FN: {false_negative}, TN: {true_negative}")

# Nakoniec vypočítame naše metriky
if true_positive == 0:
    precision = 0
    recall = 0
    f1 = 0
else:
    precision = true_positive / (true_positive + false_positive)  # Presnosť
    recall = true_positive / (true_positive + false_negative)  # Návratnosť
    f1 = 2 * precision * recall / (precision + recall)  # F1
accuracy = (true_positive + true_negative) / size_test  # Správnosť

print(f"precision: {precision}")
print(f"accuracy: {accuracy}")
print(f"recall: {recall}")
print(f"f1: {f1}")
