import numpy.random
from random import choices
from random import randint

import src.Configuration


def setMutation(configuration: src.Configuration.Composer):

    def custom_mutation(offspring, ga_instance):
        """Mutazione di un singolo gene/elemento musicale per individuo/melodia"""
        num_individuals = offspring.shape[0]
        num_genes = offspring.shape[1]
        mutations = ["pitchUp", "pitchDown", "sustain", "rest"]
        weights = [configuration.weight_pitchUp, configuration.weight_pitchDown, configuration.weight_sustain, configuration.weight_rest]
        samples = choices(mutations, weights, k = num_individuals) #genera k scelte
        for individual_idx in range(num_individuals):
            random_gene_idx = numpy.random.choice(range(num_genes))
            gene = offspring[individual_idx, random_gene_idx]
            type = samples[individual_idx]

            if type == "pitchUp":
                if gene == configuration.sustain_value:
                    i = 1
                    while(gene == configuration.sustain_value):
                        gene = offspring[individual_idx, random_gene_idx-i]
                        i += 1
                if gene == configuration.rest_value:
                    gene = randint(configuration.rest_value + 1, configuration.sustain_value - 1) #visto che il gene è una pausa, viene scelta una nota casuale

                if gene + 1 == configuration.sustain_value:
                    gene -= 1 #se non ci sono note con frequenza più alta disponibili, la frequenza viene abbassata
                else:
                    gene += 1

            if type == "pitchDown":
                if gene == configuration.sustain_value:
                    i = 1
                    while(gene == configuration.sustain_value):
                        gene = offspring[individual_idx, random_gene_idx -i]
                        i += 1
                if gene == configuration.rest_value:
                    gene = randint(configuration.rest_value + 1, configuration.sustain_value - 1) #visto che il gene è una pausa, viene scelta una nota casuale

                if gene - 1 == configuration.rest_value:
                    gene +=1 #se non ci sono note con frequenza più bassa disponibili, la frequenza viene aumentata
                else:
                    gene -= 1

            if type == "sustain":
                gene = configuration.sustain_value

            if type == "rest":
                gene = configuration.rest_value

            offspring[individual_idx, random_gene_idx] = gene

        return offspring

    return custom_mutation



