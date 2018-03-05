import experiment


class NoModel(experiment.Model):
    def build(self, data, sess):
        self.logger.info('No model...')


component = NoModel
