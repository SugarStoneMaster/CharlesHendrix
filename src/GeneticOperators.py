import numpy.random
from random import choices
from random import randint

#mutation of a single note/gene for composition/individual
import src.Configuration


def setMutation(configuration: src.Configuration.ComposerConfig):

    def custom_mutation(offspring, ga_instance):
        num_individuals = offspring.shape[0]
        num_genes = offspring.shape[1]
        mutations = ["pitchUp", "pitchDown", "sustain", "break"]
        weights = [configuration.weight_pitchUp, configuration.weight_pitchDown, configuration.weight_sustain, configuration.weight_break]
        samples = choices(mutations, weights, k = num_individuals) #generates k choices
        for individual_idx in range(num_individuals):
            random_gene_idx = numpy.random.choice(range(num_genes))
            gene = offspring[individual_idx, random_gene_idx]
            type = samples[individual_idx]

            if type == "pitchUp":
                if gene == configuration.repeat_value:
                    i = 1
                    while(gene == configuration.repeat_value):
                        gene = offspring[individual_idx, random_gene_idx-i]
                        i += 1
                if gene == configuration.break_value:
                    gene = randint(configuration.break_value+1, configuration.repeat_value-1)

                if gene + 1 == configuration.repeat_value:
                    gene -= 1
                else:
                    gene += 1

            if type == "pitchDown":
                if gene == configuration.repeat_value:
                    i = 1
                    while(gene == configuration.repeat_value):
                        gene = offspring[individual_idx, random_gene_idx -i]
                        i += 1
                if gene == configuration.break_value:
                    gene = randint(configuration.break_value+1, configuration.repeat_value-1)

                if gene - 1 == configuration.break_value:
                    gene +=1
                else:
                    gene -= 1

            if type == "sustain":
                gene = configuration.repeat_value

            if type == "break":
                gene = configuration.break_value

            offspring[individual_idx, random_gene_idx] = gene

        return offspring

    return custom_mutation



