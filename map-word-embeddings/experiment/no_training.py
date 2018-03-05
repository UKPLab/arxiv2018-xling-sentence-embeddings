import experiment


class NoTraining(experiment.Training):
    def start(self, model, data, sess, evaluation):
        self.logger.info('No training...')


component = NoTraining
