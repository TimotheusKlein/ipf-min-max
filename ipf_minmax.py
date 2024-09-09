## ipf with bandwith margins
## The MIT License (MIT)
## https://opensource.org/license/MIT

## Copyright (c) Mon Sep 09 2024 TKMG
## www.tkmg.be 

## This function fits a given numpy <matrix> to margin vectors 
## provided with bandwidths. These min/max vectors <row_min>, 
## <row_max>, <col_min>, <col_max> are provided as numpy arrays.
## To stop the fitting process, a maximum <gap> between matrix
## row and column sums and the respective bandwidths must be
## given, a minimum <gap_improvement> between iterations and a 
## maximum number of <iterations>.

## The intended application is calculating the trip distribution
## in aggregated transport demand models where demand production
## or attraction or both are/is flexible. Also, if there are
## only few demand origins and widespread attraction resp.
## destination potential, the probability distribution given
## in the matrix can get totally obliterated when using fixed
## margin vectors.

## The numeric example is taken from C. Schiller, "Auslastungs-
## abhängige Attraktivitäten in makroskopischen Zielwahlmodellen",
## Straßenverkehrstechnik 5.2010
## https://www.researchgate.net/publication/225005713_Auslastungsabhangige_Attraktivitaten_in_makroskopischen_Zielwahlmodellen


import numpy as np

def ipfminmax(matrix, row_min, row_max, col_min, col_max, gap, gap_improvement, iterations):
    # setup gap monitoring
    mgap = 0
    no_gap_improvement = 0
    
    # margin vectors bandwith
    r_min = np.min([row_min, row_max],axis=0)
    r_max = np.max([row_min, row_max],axis=0)
    c_min = np.min([col_min, col_max],axis=0)
    c_max = np.max([col_min, col_max],axis=0)

    # start iteration
    for it in range(iterations):
        
        # margin vectors
        rowsum = np.sum(matrix,axis=1)
        colsum = np.sum(matrix,axis=0)
        
        # correction factor vectors
        fr = np.ones(rowsum.shape)
        fr = np.where(r_min > rowsum,np.where(rowsum > 0,r_min/rowsum,fr),fr)
        fr = np.where(r_max < rowsum,np.where(rowsum > 0,r_max/rowsum,fr),fr)
        fc = np.ones(colsum.shape)
        fc = np.where(c_min > colsum,np.where(rowsum > 0,c_min/colsum,fc),fc)
        fc = np.where(c_max < colsum,np.where(rowsum > 0,c_max/colsum,fc),fc)

        # current gap
        r_gap = np.sum(np.min([np.where(fr != 1.0, np.absolute(rowsum - r_min), 0),
                               np.where(fr != 1.0, np.absolute(rowsum - r_max), 0)],axis=0))
        
        c_gap = np.sum(np.min([np.where(fc != 1.0, np.absolute(colsum - c_min), 0),
                               np.where(fc != 1.0, np.absolute(colsum - c_max), 0)],axis=0))
        ngap = r_gap + c_gap
        if abs(ngap-mgap) < gap_improvement: no_gap_improvement += 1
        if ngap < gap or no_gap_improvement == 10:
            break
        else:
            mgap = ngap + 0.0
            
        # matrix fitting
        matrix = matrix * fc
        matrix = (matrix.T * fr).T
        
    return matrix, it+1, ngap, mgap-ngap


# example from: C. Schiller SVT 5.2010

m = np.array([[0.00, 0.99, 1.00, 0.98, 0.98],
              [0.99, 0.00, 0.96, 0.92, 0.73],
              [1.00, 0.96, 0.00, 0.99, 0.93],
              [0.98, 0.92, 0.99, 0.00, 0.88],
              [0.98, 0.73, 0.93, 0.88, 0.00]])

r_min = np.array([50, 100, 50, 100, 200])
r_max = np.array([50, 100, 50, 100, 200])
c_min = np.array([0, 0, 0, 0, 0])
c_max = np.array([150, 60, 175, 175, 100])


print(m)
print('r_min:\n', r_min)
print('r_max:\n', r_max)
print('c_min:\n', c_min)
print('c_max:\n', c_max)

print(ipfminmax(m, r_min, r_max, c_min, c_max, 0.001, 0.00001, 25))

    
    
