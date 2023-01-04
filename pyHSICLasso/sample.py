#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

# from pyHSICLasso import HSICLasso
# from api import HSICLasso
from pyHSICLasso import *
import numpy as np
standard_library.install_aliases()


def main():
    hsic_lasso = HSICLasso()
    X = np.random.randn(200,50)
    Y = np.random.randn(200,)
    # hsic_lasso.input("user_data.csv", output_list=['c1', 'c2', 'c3', 'c4', 'c5,', 'c6', 'c7', 'c8', 'c9', 'c10'])
    hsic_lasso.input(X,Y)
        # ,'c11', 'c12', 'c13', 'c14', 'c15,', 'c16', 'c17', 'c18', 'c19', 'c20','c21', 'c22', 'c23', 'c24', 'c25,', 'c26', 'c27', 'c28', 'c29', 'c30'])
    hsic_lasso.regression(3)
    hsic_lasso.dump()
    select_index = hsic_lasso.get_index()
    print(select_index)
    print(hsic_lasso.get_index_score())
    hsic_lasso.plot_path()
    print(hsic_lasso.get_features())
    X_select = hsic_lasso.X_in[select_index, :]
    np.savetxt('X_select.txt', X_select, fmt='%d', encoding='utf-8')


if __name__ == "__main__":
    main()
