package neuralNetwork

class ResultHandler {

    var nSuccess: Int = 0
        private set
    private val threshold = 0.5

    fun countSuccess(calcForwardOutput: Float, target: Float) {
        if (calcForwardOutput < threshold && target == 0f || calcForwardOutput >= 1 - threshold && target == 1f) {
            this.nSuccess += 1
        }
    }

    fun parseResult(result: Float): Int {
        return if (result < threshold) 0 else 1
    }

    fun resetNSuccess() {
        this.nSuccess = 0
    }
}
