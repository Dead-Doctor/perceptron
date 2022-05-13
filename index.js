import assert from 'assert';
import { writeFileSync, appendFileSync } from 'fs';

function randomRange(min, max) {
    assert(min < max);
    return Math.floor(Math.random() * (max - min - 1)) + min;
}

function valueToColor(value, expectedRange) {
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
    saveDebug(name, expectedRange) {
        const fileName = name + '.ppm';
        writeFileSync(fileName, `P6 ${this.size} ${this.size} 255\n`);

        let data = [];
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                data.push(...valueToColor(this.grid[y][x], expectedRange));
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
        ////// Random Rect 1 //////
        const x = randomRange(0, this.size / 2);
        const y = randomRange(0, this.size / 2);
        const width = randomRange(this.size / 2 - x, this.size - x);
        const height = randomRange(this.size / 2 - y, this.size - y);
        ////// Random Rect 2 //////
        // const width = randomRange(1, this.size);
        // const height = randomRange(1, this.size);
        // const x = randomRange(0, this.size - width);
        // const y = randomRange(0, this.size - height);
        ///////////////////////////
        this.fillRect(x, y, width, height, 1);
    }

    rngCircle() {
        ////// Random Circle 1 //////
        const x = randomRange(this.size / 4, (this.size / 4) * 3);
        const y = randomRange(this.size / 4, (this.size / 4) * 3);
        const radius = randomRange(this.size / 4 - 1, Math.min(x, y, this.size - x, this.size - y));
        ////// Random Circle 2 //////
        // const radius = randomRange(1, this.size / 2);
        // const x = randomRange(radius, this.size - radius);
        // const y = randomRange(radius, this.size - radius);
        /////////////////////////////
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
const record = true;
const recordFrames = 1_200;

const totalShapes = 100_000;
const size = 64;
const weights = new Layer(size);

let frames = '';
let frameTime = 2.82;

function recordFrame(i, layer) {
    frameTime -= Math.pow(1.045, -i * 10);

    const name = `video/frame-${i.toString().padStart(4, '0')}`;
    layer.saveDebug(name, 10);
    frames += `file '${name}.ppm'\n`;
    frames += `duration ${frameTime}\n`;
}

// Train the network -----------------------------------------------------------------------------------------------
for (let i = 0; i < totalShapes; i++) {
    if (record && i < recordFrames) recordFrame(i, weights);

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

weights.saveDebug('trainedWeights', 32);

if (record) writeFileSync('frames.txt', frames);

// Test the network -------------------------------------------------------------------------------------------------

const totalTests = 1_000;

let correctRectTests = 0;
let correctCircleTests = 0;
const totalRectTestLayer = new Layer(size);
const totalCircleTestLayer = new Layer(size);

for (let i = 0; i < totalTests; i++) {
    if (showLogs) console.log(`${i}. Test`);

    // Rectangle
    const rectTestLayer = new Layer(size);
    rectTestLayer.rngRect();
    totalRectTestLayer.addInput(rectTestLayer);
    let rectTestOutput = weights.feedForward(rectTestLayer);
    if (rectTestOutput > 0) correctRectTests++;

    // Circle
    const circleTestLayer = new Layer(size);
    circleTestLayer.rngCircle();
    totalCircleTestLayer.addInput(circleTestLayer);
    let circleTestOutput = weights.feedForward(circleTestLayer);
    if (circleTestOutput > 0) correctCircleTests++;
}

totalRectTestLayer.saveDebug('rectTest', 1000);
totalCircleTestLayer.saveDebug('circleTest', 1000);

console.log(`Rectangle Test Output: ${Math.round((correctRectTests / totalTests) * 1000) / 10}%`);
console.log(`Circle    Test Output: ${Math.round((correctCircleTests / totalTests) * 1000) / 10}%`);
