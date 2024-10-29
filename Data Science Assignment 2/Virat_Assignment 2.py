{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import all the necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### I - Virat Kohli Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"virat.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Runs</th>\n",
       "      <th>Mins</th>\n",
       "      <th>BF</th>\n",
       "      <th>4s</th>\n",
       "      <th>6s</th>\n",
       "      <th>SR</th>\n",
       "      <th>Pos</th>\n",
       "      <th>Dismissal</th>\n",
       "      <th>Inns</th>\n",
       "      <th>Opposition</th>\n",
       "      <th>Ground</th>\n",
       "      <th>Start Date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12</td>\n",
       "      <td>33</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>54.54</td>\n",
       "      <td>2</td>\n",
       "      <td>lbw</td>\n",
       "      <td>1</td>\n",
       "      <td>v Sri Lanka</td>\n",
       "      <td>Dambulla</td>\n",
       "      <td>18-Aug-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>82</td>\n",
       "      <td>67</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>55.22</td>\n",
       "      <td>2</td>\n",
       "      <td>caught</td>\n",
       "      <td>2</td>\n",
       "      <td>v Sri Lanka</td>\n",
       "      <td>Dambulla</td>\n",
       "      <td>20-Aug-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>25</td>\n",
       "      <td>40</td>\n",
       "      <td>38</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>65.78</td>\n",
       "      <td>1</td>\n",
       "      <td>run out</td>\n",
       "      <td>1</td>\n",
       "      <td>v Sri Lanka</td>\n",
       "      <td>Colombo (RPS)</td>\n",
       "      <td>24-Aug-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>54</td>\n",
       "      <td>87</td>\n",
       "      <td>66</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>81.81</td>\n",
       "      <td>1</td>\n",
       "      <td>bowled</td>\n",
       "      <td>1</td>\n",
       "      <td>v Sri Lanka</td>\n",
       "      <td>Colombo (RPS)</td>\n",
       "      <td>27-Aug-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>31</td>\n",
       "      <td>45</td>\n",
       "      <td>46</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>67.39</td>\n",
       "      <td>1</td>\n",
       "      <td>lbw</td>\n",
       "      <td>2</td>\n",
       "      <td>v Sri Lanka</td>\n",
       "      <td>Colombo (RPS)</td>\n",
       "      <td>29-Aug-08</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Runs Mins  BF  4s  6s     SR  Pos Dismissal  Inns   Opposition  \\\n",
       "0   12   33  22   1   0  54.54    2       lbw     1  v Sri Lanka   \n",
       "1   37   82  67   6   0  55.22    2    caught     2  v Sri Lanka   \n",
       "2   25   40  38   4   0  65.78    1   run out     1  v Sri Lanka   \n",
       "3   54   87  66   7   0  81.81    1    bowled     1  v Sri Lanka   \n",
       "4   31   45  46   3   1  67.39    1       lbw     2  v Sri Lanka   \n",
       "\n",
       "          Ground Start Date  \n",
       "0       Dambulla  18-Aug-08  \n",
       "1       Dambulla  20-Aug-08  \n",
       "2  Colombo (RPS)  24-Aug-08  \n",
       "3  Colombo (RPS)  27-Aug-08  \n",
       "4  Colombo (RPS)  29-Aug-08  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 132 entries, 0 to 131\n",
      "Data columns (total 12 columns):\n",
      " #   Column      Non-Null Count  Dtype \n",
      "---  ------      --------------  ----- \n",
      " 0   Runs        132 non-null    object\n",
      " 1   Mins        132 non-null    object\n",
      " 2   BF          132 non-null    int64 \n",
      " 3   4s          132 non-null    int64 \n",
      " 4   6s          132 non-null    int64 \n",
      " 5   SR          132 non-null    object\n",
      " 6   Pos         132 non-null    int64 \n",
      " 7   Dismissal   132 non-null    object\n",
      " 8   Inns        132 non-null    int64 \n",
      " 9   Opposition  132 non-null    object\n",
      " 10  Ground      132 non-null    object\n",
      " 11  Start Date  132 non-null    object\n",
      "dtypes: int64(5), object(7)\n",
      "memory usage: 12.5+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Runs          0\n",
       "Mins          0\n",
       "BF            0\n",
       "4s            0\n",
       "6s            0\n",
       "SR            0\n",
       "Pos           0\n",
       "Dismissal     0\n",
       "Inns          0\n",
       "Opposition    0\n",
       "Ground        0\n",
       "Start Date    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Spread in Runs\n",
    "Question 1: Analyse the spread of Runs scored by Virat in all his matches and report the difference between the scores at the 50th percentile and the 25th percentile respectively.\n",
    "\n",
    "    a)16.5\n",
    "    b)22.5\n",
    "    c)26.5\n",
    "    d)32.5\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Runs'] = df['Runs'].apply(lambda x:int(x[:-1]) if x[-1] == '*' else int(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    132.000000\n",
       "mean      46.848485\n",
       "std       41.994635\n",
       "min        0.000000\n",
       "25%       10.000000\n",
       "50%       32.500000\n",
       "75%       80.250000\n",
       "max      154.000000\n",
       "Name: Runs, dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Runs'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22.5"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = 32.5 - 10\n",
    "a"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Box Plots\n",
    "\n",
    "Question 2: Plot a Box Plot to analyse the spread of Runs that Virat has scored. The upper fence in the box plot lies in which interval?\n",
    "\n",
    "    a)100-120\n",
    "    b)120-140\n",
    "    c)140-160\n",
    "    d)160-180\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.boxplot(df.Runs)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### False Statement\n",
    "\n",
    "Q3:Consider the following statements and choose the correct option\n",
    "\n",
    "     I - Virat has played the maximum number of matches in 2011\n",
    "     II - Virat has the highest run average in the year 2017\n",
    "     III - Virat has the maximum score in a single match and the highest run average in the year 2016.\n",
    "\n",
    "Which of the above statements is/are false?\n",
    "\n",
    "    a)I and II\n",
    "    b)I and III\n",
    "    c)II\n",
    "    d)III\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#  I - Virat has played the maximum number of matches in 2011\n",
    "df['Start Date'] = df['Start Date'].apply(lambda x: (x[-2:]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11    31\n",
       "13    23\n",
       "14    17\n",
       "10    16\n",
       "12    11\n",
       "15    10\n",
       "16    10\n",
       "09     6\n",
       "08     5\n",
       "17     3\n",
       "Name: Start Date, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Start Date'].value_counts() # Statement I is correct "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Start Date</th>\n",
       "      <th>08</th>\n",
       "      <th>09</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "      <th>17</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Runs</th>\n",
       "      <td>31.8</td>\n",
       "      <td>38.333333</td>\n",
       "      <td>45.375</td>\n",
       "      <td>42.0</td>\n",
       "      <td>40.363636</td>\n",
       "      <td>47.826087</td>\n",
       "      <td>58.529412</td>\n",
       "      <td>30.4</td>\n",
       "      <td>73.9</td>\n",
       "      <td>61.666667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Start Date    08         09      10    11         12         13         14  \\\n",
       "Runs        31.8  38.333333  45.375  42.0  40.363636  47.826087  58.529412   \n",
       "\n",
       "Start Date    15    16         17  \n",
       "Runs        30.4  73.9  61.666667  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#II - Virat has the highest run average in the year 2017\n",
    "pd.pivot_table(df, values = 'Runs', columns = ['Start Date'], aggfunc = np.mean)\n",
    "# statement II is incorrect "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Start Date</th>\n",
       "      <th>08</th>\n",
       "      <th>09</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "      <th>17</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Runs</th>\n",
       "      <td>54</td>\n",
       "      <td>107</td>\n",
       "      <td>118</td>\n",
       "      <td>117</td>\n",
       "      <td>128</td>\n",
       "      <td>115</td>\n",
       "      <td>139</td>\n",
       "      <td>138</td>\n",
       "      <td>154</td>\n",
       "      <td>122</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Start Date  08   09   10   11   12   13   14   15   16   17\n",
       "Runs        54  107  118  117  128  115  139  138  154  122"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# III - Virat has the maximum score in a single match and the highest run average in the year 2016.\n",
    "pd.pivot_table(df, values = 'Runs', columns = ['Start Date'], aggfunc = np.max)\n",
    "\n",
    "# Statement III is correct "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Option B is correct "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Maximum Frequency\n",
    "\n",
    "Q4:Plot a histogram for the Mins column with 15 bins. Among the three ranges mentioned below, which one has the highest frequency?\n",
    "\n",
    "A - [54.6,68)\n",
    "\n",
    "B - [68,81.4)\n",
    "\n",
    "C - [121.6,135)\n",
    "\n",
    "    a)A - [54.6,68)\n",
    "    b)B - [68,81.4)\n",
    "    c)C - [121.6,135)\n",
    "    d)All the bin ranges have the same frequency\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = df[-(df['Mins'] == '-')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_925/3080766231.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2['Mins'] = df2['Mins'].apply(lambda x: int(x))\n"
     ]
    }
   ],
   "source": [
    "df2['Mins'] = df2['Mins'].apply(lambda x: int(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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+OCDD3Kdximd7vzefffdeOWVV+J73/tevPLKK/HEE0/Ef//3f8ctt9wyYN+FCxeWzekPf/jDHMMflDPNYcSZn5PVOocRUXZe+/fvjx//+MdRKBTitttuK9tvNM7hYN4PRv1rMI1hf/mXf5nuuuuusnWf/vSn0/Lly0doRJVz4MCBFBFp48aNpXXz589Pn//850duUB/Bvffem2bOnHnSbcePH08NDQ3p/vvvL637wx/+kGpra9O//uu/ZhphZX3rW99KF110UTp+/HhKqbrnLiLSk08+WXo8mPk6dOhQGjduXFq3bl1pn//5n/9Jn/jEJ9Kzzz6bbeyD8eHzO5mf//znKSLSG2+8UVp37bXXpm9961vDO7gKOdk5nuk5Odbm8POf/3yaO3du2bpqmcMPvx9Uw2twzF75OHr0aGzbti3a2trK1re1tcXmzZtHaFSV09vbGxERkydPLlu/YcOGqKuri4svvjgWLlwYBw4cGInhnZXdu3dHU1NTtLa2xh133BGvv/56RER0d3dHT09P2VwWi8W49tprq3Iujx49Gj/5yU/iq1/9atnNE6t57v7UYOZr27ZtcezYsbJ9mpqaYvr06VU5p729vVEoFAbcl+rRRx+NKVOmxCWXXBJ333131VypO+F0z8mxNIe///3vY/369fG1r31twLZqmMMPvx9Uw2tw2O5qO9Leeuut+OCDDwbczK6+vn7ATe+qTUopli1bFldffXVMnz69tL69vT2++MUvRktLS3R3d8f3vve9mDt3bmzbtm3U/XW7D7viiivikUceiYsvvjh+//vfxz/90z/F7NmzY+fOnaX5OtlcvvHGGyMx3I/kqaeeikOHDsWCBQtK66p57j5sMPPV09MT48ePjz//8z8fsE+1vT7/8Ic/xPLly+POO+8su3HXl7/85WhtbY2Ghob49a9/HStWrIhf/vKXA+57NVqd6Tk5luZw7dq1UVNTE/PmzStbXw1zeLL3g2p4DY7Z+DjhT//PMuKPE/XhddVm8eLF8atf/SpefvnlsvW333576d/Tp0+Pyy67LFpaWmL9+vUDXlSjTXt7e+nfM2bMiKuuuiouuuiiWLt2belDbmNlLh9++OFob28vu+10Nc/dqZzNfFXbnB47dizuuOOOOH78eDz00ENl2xYuXFj69/Tp02PatGlx2WWXxSuvvBKXXnpp7qEO2dk+J6ttDiMifvzjH8eXv/zlOO+888rWV8Mcnur9IGJ0vwbH7I9dpkyZEuecc86Agjtw4MCAGqwmS5YsiaeffjpefPHFmDp16mn3bWxsjJaWlti9e3em0VXOxIkTY8aMGbF79+7Sb72Mhbl844034vnnn4+vf/3rp92vmuduMPPV0NAQR48ejbfffvuU+4x2x44di7/5m7+J7u7u6OrqOuPtyi+99NIYN25cVc5pxMDn5FiYw4iIl156KXbt2nXG12TE6JvDU70fVMNrcMzGx/jx42PWrFkDLo91dXXF7NmzR2hUZy+lFIsXL44nnngiXnjhhWhtbT3j1xw8eDD27dsXjY2NGUZYWf39/fHb3/42GhsbS5c9/3Qujx49Ghs3bqy6uVyzZk3U1dXFjTfeeNr9qnnuBjNfs2bNinHjxpXts3///vj1r39dFXN6Ijx2794dzz//fHzyk58849fs3Lkzjh07VpVzGjHwOVntc3jCww8/HLNmzYqZM2eecd/RModnej+oitfgsH+kdQStW7cujRs3Lj388MPpN7/5TVq6dGmaOHFi2rNnz0gPbcj+7u/+LtXW1qYNGzak/fv3l5Z33303pZTS4cOH07e//e20efPm1N3dnV588cV01VVXpb/4i79IfX19Izz6M/v2t7+dNmzYkF5//fW0ZcuWdNNNN6WamprSXN1///2ptrY2PfHEE2nHjh3pS1/6UmpsbKyKczvhgw8+SBdccEH67ne/W7a+Gufu8OHDafv27Wn79u0pItLKlSvT9u3bS7/tMZj5uuuuu9LUqVPT888/n1555ZU0d+7cNHPmzPT++++P1GmVnO78jh07lm655ZY0derU9Oqrr5a9Hvv7+1NKKb322mvp+9//ftq6dWvq7u5O69evT5/+9KfTZz/72VFxfimd/hwH+5ys1jk8obe3N51//vlp9erVA75+NM/hmd4PUhr9r8ExHR8ppfQv//IvqaWlJY0fPz5deumlZb+aWk0i4qTLmjVrUkopvfvuu6mtrS196lOfSuPGjUsXXHBBmj9/ftq7d+/IDnyQbr/99tTY2JjGjRuXmpqa0rx589LOnTtL248fP57uvffe1NDQkIrFYrrmmmvSjh07RnDEQ/fcc8+liEi7du0qW1+Nc/fiiy+e9Pk4f/78lNLg5uu9995LixcvTpMnT04TJkxIN91006g559OdX3d39ylfjy+++GJKKaW9e/ema665Jk2ePDmNHz8+XXTRRenv//7v08GDB0f2xP7E6c5xsM/Jap3DE374wx+mCRMmpEOHDg34+tE8h2d6P0hp9L8GC//vRAAAshizn/kAAEYn8QEAZCU+AICsxAcAkJX4AACyEh8AQFbiAwDISnwAAFmJDwAgK/EBAGQlPgCArMQHAJDV/wGShrSpRS38BAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df2.Mins, bins = 15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option C is the correct "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "anaconda-panel-2023.05-py310",
   "language": "python",
   "name": "conda-env-anaconda-panel-2023.05-py310-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
