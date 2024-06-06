def get_equation(feature):

    featDict = {
        'skewness': ('A measure of symmetry across the sample mean [11]',
                     r'''
                     $$
                     \frac{\frac{1}{N}\displaystyle\sum_{n=1}^N (x_n - \overline{x})^3}{\sqrt{\frac{1}{N}\displaystyle\sum_{n=1}^N (x_n - \overline{x})^2}^3}
                     $$
                     '''
                    ),
        'kurtosis': ('A measure of tail heaviness of the normal distribution. A heavier tail has more outliers [11].',
                     r'''
                     $$
                     \frac{\frac{1}{N}\displaystyle\sum_{n=1}^N (x_n - \overline{x})^4}{\sqrt{\frac{1}{N}\displaystyle\sum_{n=1}^N (x_n - \overline{x})^2}^4}
                     $$
                     '''
                     ),
        'crest': ('The peak to RMS ratio of a waveform [12].',
                  r'''
                    $$
                    \frac{max|x_n|}{\sqrt{\frac{1}{N}\displaystyle\sum_{n=1}^N x_n^2}}
                    $$
                 '''
                 ),
        'shape': ('Parameter that affects the general shape of a distribution [13].',
                  r'''
                   $$
                   \frac{\sqrt{\frac{1}{N}\displaystyle\sum_{n=1}^N x_n^2}}{\frac{1}{N}\displaystyle\sum_{n=1}^N |x_n|}
                   $$
                  ''' ),
        'impulse': ('Height of peak to the mean signal level [14].',
                    r'''
                    $$
                    \frac{max|x_n|}{\frac{1}{N}\displaystyle\sum_{n=1}^N |x_n|}
                    $$
                    '''),
        'margin': ('Peak amplitude to squared mean of squared roots of absolute amplitudes – also called “clearance factor” [14]',
                   r'''
                   $$
                   \frac{max|x_n|}{{\frac{1}{N}\displaystyle\sum_{n=1}^N \sqrt{|x_n|}}^2}
                   $$
                   ''')
    }

    return featDict.get(feature)