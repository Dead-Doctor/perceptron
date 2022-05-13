import assert from 'assert';
import { writeFileSync, appendFileSync } from 'fs';

function randomRange(min, max) {
    assert(min < max);
    return Math.floor(Math.random() * (max - min - 1)) + min;
}

function valueToColor(value) {
    const expectedRange = 32;

    // normalize values using sigmoid function to -255 to 255
    let absolute = 255 / (1 + Math.exp(-value / expectedRange));

    return [absolute, absolute, 0xff - absolute];
}

function testPPM() {
    writeFileSync('test.ppm', `P6 512 512 255\n`);
    data = [];
    for (let i = -256; i < 256; i++) {
        for (let j = -256; j < 256; j++) {
            data.push(...valueToColor(i));
        }
    }
    appendFileSync('test.ppm', Buffer.from(data));
}

class Layer {
    grid = [];

    constructor(size) {
        this.size = size;
        for (let y = 0; y < size; y++) {
            this.grid[y] = [];
            for (let x = 0; x < size; x++) {
                this.grid[y][x] = 0;
            }
        }
    }

    // Save im PPM format
    saveDebug(name) {
        const fileName = name + '.ppm';
        writeFileSync(fileName, `P6 ${this.size} ${this.size} 255\n`);

        let data = [];
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                data.push(...valueToColor(this.grid[y][x]));
            }
        }
        appendFileSync(fileName, Buffer.from(data));
    }

    fillRect(x, y, width, height, value) {
        assert(x >= 0 && x < this.size);
        assert(y >= 0 && y < this.size);
        assert(width > 0 && x + width <= this.size);
        assert(height > 0 && y + height <= this.size);

        for (let currentY = y; currentY <= y + height; currentY++) {
            for (let currentX = x; currentX <= x + width; currentX++) {
                this.grid[currentY][currentX] += value;
            }
        }
    }

    fillCircle(x, y, radius, value) {
        assert(x >= 0 && x < this.size);
        assert(y >= 0 && y < this.size);
        assert(radius > 0 && x + radius <= this.size);
        assert(radius > 0 && y + radius <= this.size);

        for (let currentY = y - radius; currentY <= y + radius; currentY++) {
            for (let currentX = x - radius; currentX <= x + radius; currentX++) {
                if ((currentX - x) * (currentX - x) + (currentY - y) * (currentY - y) <= radius * radius) {
                    this.grid[currentY][currentX] += value;
                }
            }
        }
    }

    rngRect() {
        const x = randomRange(0, this.size / 2);
        const y = randomRange(0, this.size / 2);
        const width = randomRange(this.size / 2 - x, this.size - x);
        const height = randomRange(this.size / 2 - y, this.size - y);
        this.fillRect(x, y, width, height, 1);
    }

    rngCircle() {
        const x = randomRange(this.size / 4, (this.size / 4) * 3);
        const y = randomRange(this.size / 4, (this.size / 4) * 3);
        const radius = randomRange(this.size / 4 - 1, Math.min(x, y, this.size - x, this.size - y));
        this.fillCircle(x, y, radius, -1);
    }

    feedForward(input) {
        let output = 0.0;
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                output += this.grid[y][x] * input.grid[y][x];
            }
        }
        return output;
    }

    addInput(input) {
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                this.grid[y][x] += input.grid[y][x];
            }
        }
    }
}

const showLogs = false;

const totalShapes = 100_000;
const size = 64;
const weights = new Layer(size);

// Train the network -----------------------------------------------------------------------------------------------
for (let i = 0; i < totalShapes; i++) {
    if (showLogs) console.log(`${i}. Input`);

    // Rectangle
    let input = new Layer(size);
    input.rngRect();
    let output = weights.feedForward(input);
    if (output <= 0) weights.addInput(input);

    // Circle
    input = new Layer(size);
    input.rngCircle();
    output = weights.feedForward(input);
    if (output <= 0) weights.addInput(input);
}

weights.saveDebug('trainedWeights');

// Test the network -------------------------------------------------------------------------------------------------

const totalTests = 1_000;

let avgRectTestOutput = 0;
let avgCircleTestOutput = 0;
const totalRectTestLayer = new Layer(size);
const totalCircleTestLayer = new Layer(size);

for (let i = 0; i < totalTests; i++) {
    if (showLogs) console.log(`${i}. Test`);

    // Rectangle
    const rectTestLayer = new Layer(size);
    rectTestLayer.rngRect();
    totalRectTestLayer.addInput(rectTestLayer);
    let rectTestOutput = weights.feedForward(rectTestLayer);
    avgRectTestOutput += rectTestOutput;

    // Circle
    const circleTestLayer = new Layer(size);
    circleTestLayer.rngCircle();
    totalCircleTestLayer.addInput(circleTestLayer);
    let circleTestOutput = weights.feedForward(circleTestLayer);
    avgCircleTestOutput += circleTestOutput;
}

totalRectTestLayer.saveDebug('rectTest');
totalCircleTestLayer.saveDebug('circleTest');

console.log(`Rect Test Output: ${avgRectTestOutput / totalTests}`);
console.log(`Circle Test Output: ${avgCircleTestOutput / totalTests}`);