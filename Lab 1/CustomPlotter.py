import matplotlib.pyplot as plt
import itertools
import logging
import logging.config
import os
import json


class CustomPlotter(Exception):

    def __init__(self):
        self.colorStyle = itertools.cycle(('b', 'g', 'r', 'y', 'k'))
        self.markerStyle = itertools.cycle(('.', ',',  '+',  ',','2','1'))
        self.lineStyle = itertools.cycle(('-', '--', '-.', ':'))

        self.logger = logging.getLogger(__name__)
        path = ('log_config.json')
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)

        else:
            logging.config.dictConfig({
                'version': 1,
                'disable_existing_loggers': False,  # this fixes the problem
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    },
                },
                'handlers': {
                    'default': {
                        'level': 'INFO',
                        'class': 'logging.StreamHandler',
                    },
                },
                'loggers': {
                    '': {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': True
                    }
                }
            })


    def graph_plot(self, yaxis, xaxis, x_label, y_label, title):
        plt.figure (figsize=(8, 5))
        plt.scatter (xaxis, yaxis, color="red")
        plt.plot (xaxis, xaxis)
        plt.title (title)
        plt.xlabel (x_label)
        plt.ylabel (y_label)
        plt.grid (True)
        return


    def weight_graph_plot(self,x, x_label, y_label, title, weight=1):
        plt.figure (figsize=(8, 5))
        if weight == 1:
            plt.plot (x, 'ro')
        else:
            plt.plot (x)
        plt.xlabel (x_label)
        plt.ylabel (y_label)
        plt.title (title)
        #    plt.legend(loc="best")
        plt.grid (True)
        return

    def multiplot_sameX(self, title, xlabel, ylabel, **yaxisTitleAndValue):

        try:
            self.logger.debug("Reading the input args in multiplot_sameX")
            if yaxisTitleAndValue is not None:
                for key, value in yaxisTitleAndValue.items():
                    self.logger.debug("plotting for key:" + str(key) + " with Value")
                    plt.plot(value, marker=next(self.markerStyle), linestyle=next(self.lineStyle),
                             color=next(self.colorStyle), label=key)
        except (SystemExit, KeyboardInterrupt):
            raise

        except Exception as e:
            self.logger.error('Failed to plot value', exc_info=True)
            raise

        plt.title(title)
        plt.legend(loc="lower right")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def multiplot_withXvalues(self, title, xlabel, ylabel, *xAxisValue, **yaxisTitleAndValue):
        x_values = []
        try:
            self.logger.debug("Reading the input args in multiplot_withXvalues")
            for x_value in xAxisValue:
                x_values.append(x_value)
            #self.logger.debug ("x_values=",x_values)

            if yaxisTitleAndValue is not None:
                i = 0

                for key, value in yaxisTitleAndValue.items():
                    self.logger.debug("plotting for key:" + str(key) + "with Value")
                    plt.plot(x_values[i], value, marker=next(self.markerStyle), linestyle=next(self.lineStyle),
                             color=next(self.colorStyle), label=key)

                    i = i + 1
        except (SystemExit, KeyboardInterrupt):
            raise

        except Exception as e:
            self.logger.error('Failed to plot value', exc_info=True)
            raise

        plt.title(title)
        plt.legend(loc="best")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig ("Output/" + title+".png")
        plt.show()

    def multiplot_subplots_withoutXaxis(self, title,sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, **yaxisTitleAndValue):
        titles = []
        values = []
        try:
            self.logger.debug("Reading the input args in multiplot_subplots_withoutXaxis")
            if yaxisTitleAndValue is not None:
                for title, value in yaxisTitleAndValue.items():
                    titles.append(title)
                    values.append(value)

            i = 0
            fig = plt.figure('Hist')
            Position = range(1, (sp_rows * sp_cols) + 1)
            for row in range(sp_rows):
                for col in range(sp_cols):
                    ax = fig.add_subplot(sp_rows, sp_cols, Position[i])
                    for w in range(valuesPerSubplot):
                        self.logger.debug("plotting for row:"+str(row)+"-Col:"+str(col))
                        ax.plot(values[i], marker=next(self.markerStyle), linestyle=next(self.lineStyle),
                                color=next(self.colorStyle))
                        ax.set_title(titles[i])
                        ax.set(xlabel=xlabel, ylabel=ylabel)
                        i = i + 1
        except (SystemExit, KeyboardInterrupt):
            raise

        except Exception as e:
            self.logger.error('Failed to plot value', exc_info=True)
            raise
        plt.title(title)
        plt.show()


    def multiplot_subplots_withXaxis(self, title,sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *xAxisValue,
                                     **yaxisTitleAndValue):
        titles = []
        values = []
        x_values = []
        try:
            self.logger.debug("Reading the input args in multiplot_subplots_withXaxis")
            for x_value in xAxisValue:
                x_values.append(x_value)

            if yaxisTitleAndValue is not None:
                for title, value in yaxisTitleAndValue.items():
                    titles.append(title)
                    values.append(value)

            fig = plt.figure('1')
            i = 0
            Position = range(1, (sp_rows * sp_cols) + 1)
            for row in range(sp_rows):
                for col in range(sp_cols):
                    ax = fig.add_subplot(sp_rows, sp_cols, Position[i])
                    self.logger.debug("plotting for row:" + str(row) + "-Col:" + str(col))
                    for w in range(valuesPerSubplot):
                        ax.plot(x_values[i], values[i], marker=next(self.markerStyle), linestyle=next(self.lineStyle),
                                color=next(self.colorStyle))
                        ax.set_title(titles[i])
                        ax.set(xlabel=xlabel, ylabel=ylabel)
                        i = i + 1
        except (SystemExit, KeyboardInterrupt):
            raise

        except Exception as e:
            self.logger.error('Failed to plot value', exc_info=True)
            raise

        plt.title(title)
        plt.show()
        plt.grid(True)

    def multiplot_subHists(self, title,sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue):
        titles = []
        values = []
        bin_values = []
        try:
            self.logger.debug("Reading the input args in multiplot_subHists")
            for bin in bins:
                bin_values.append(bin)

            if yaxisTitleAndValue is not None:
                for title, value in yaxisTitleAndValue.items():
                    titles.append(title)
                    values.append(value)

            fig = plt.figure('Hist')
            i = 0
            Position = range(1, (sp_rows * sp_cols) + 1)
            for row in range(sp_rows):
                for col in range(sp_cols):
                    ax = fig.add_subplot(sp_rows, sp_cols, Position[i])
                    self.logger.debug("plotting for row:" + str(row) + "-Col:" + str(col))
                    for w in range(valuesPerSubplot):
                        ax.hist(values[i], bins=bin_values[i],normed=1, facecolor='g')
                        ax.set_title(titles[i])
                        ax.set(xlabel=xlabel, ylabel=ylabel)
                        i = i + 1



        except (SystemExit, KeyboardInterrupt):
            raise

        except Exception as e:
            self.logger.error('Failed to plot value', exc_info=True)
            raise
        plt.title(title)
        plt.show()

