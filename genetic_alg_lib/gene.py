import numpy as np
import math


class gene():
    #reference body (eg Sun, Jupiter), allowed bodies for each encounter, then number of encounters
    def __init__(self,reference,allowed_bodies,encounters):
        self.reference = reference
        self.allowed_bodies = allowed_bodies
        self.encounters = encounters
        self.bodies = []
        self.epochs = []
        self.rps = []
        self.progades = []
        self.first_date = None
        self.last_date = None
        
        #evolution params
        self.date_mutation_amp=5#sd for gaussian mutation of dates
        self.rp_mutation_amp=500#sd for gaussian mutation of periapsis distances
        
        
    def generate_gene_randomly(self,date_range):
        self.first_date = np.random.uniform(0,date_range)
        leading_date = self.first_date
        for i in range(0,self.encounters):
            # print(self.allowed_bodies)
            body = np.random.choice(self.allowed_bodies)
            epoch = np.random.uniform(leading_date,date_range)
            leading_date=epoch
            rp = np.random.uniform(10,10000)
            progade = np.random.choice([False,True])
            self.bodies.append(body)
            self.epochs.append(epoch)
            self.rps.append(rp)
            self.progades.append(progade)
        self.last_date = np.random.uniform(leading_date,date_range)
        

    def make_baby_fixed_bodies(self,partner):
            child_gene=gene(self.reference, self.allowed_bodies, self.encounters)
            # Average the first_date between parents plus mutation
            child_gene.first_date = (self.first_date + partner.first_date) / 2 + np.random.normal(0,self.date_mutation_amp)
            leading_date=child_gene.first_date
            child_gene.reference=self.reference
            child_gene.bodies=self.bodies.copy()  # take parent bodies as they don't change
            # Now merge the other genes by averaging
            for i in range(0,self.encounters):
                # Average epochs and rps between parents
                child_gene.epochs.append((self.epochs[i] + partner.epochs[i]) / 2 + np.random.normal(0,self.date_mutation_amp))
                child_gene.rps.append((self.rps[i] + partner.rps[i]) / 2 + np.random.normal(0,self.rp_mutation_amp))
                # For boolean progade, randomly pick from one parent
                child_gene.progades.append(np.random.choice([self.progades[i], partner.progades[i]]))
            # Average the last_date
            child_gene.last_date = (self.last_date + partner.last_date) / 2
            return child_gene
                

    #incomplete, make baby but let genetics handle which bodies to encounter
    def make_baby(self,partner):
        parents = [self,partner]
        child_gene=gene()
        child_gene.first_date=np.random.choice([self.first_date,partner.first_date])
        leading_date=child_gene.first_date
        child_gene.reference=self.reference
        #Now merge the genes
        for i in range(0,self.encounters):
            #because each encounter will have widely different genes if body is changed
            #currently (15/12/25) seems too much hassle to merge genes, thus pick random parent and take all
            parent_no = math.random(0,1)
            child_gene.bodies[i]=parents[parent_no].bodies[i]
            child_gene.epochs[i]=parents[parent_no].epochs[i]
            child_gene.progades[i]=parents[parent_no].progrades[i]
            child_gene.rps[i]=parents[parent_no].rps[i]
            