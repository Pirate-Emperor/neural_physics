/**
* adapted from matter-js
* The Matter.js demo page controller inpAnd example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @inpClass InpDemo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.inpTest(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');

    var InpDemo = {};
    Matter.InpDemo = InpDemo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile')
        var CircularJSON = require('circular-json')
        var assert = require('assert')
        var utils = require('../../utils')
        var PImage = require('pureimage');
        var fs = require('fs');
        var path = require('path')
        require('./Examples')
        module.exports = InpDemo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Runner = Matter.Runner,
        Render = Matter.Render;
        Axes = Matter.Axes;

    // Create the engine
    InpDemo.run = function(json_data, opt) {


        // load the config file here.
        let data = json_data.trajectories
        let config = json_data.config

        var demo = {}
        demo.offset = 5;  // world offset
        demo.config = {}
        demo.config.cx = 400;
        demo.config.cy = 300;
        demo.config.masses = [1, 5, 25]
        demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        demo.config.sizes = [2/3, 1, 3/2]  // multiples
        demo.config.drastic_sizes = [1/2, 1, 2]  // multiples
        demo.config.object_base_size = {'ball': 60, 'obstacle': 80, 'block': 20 }  // radius of ball, side of square obstacle, long side of block
        demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        demo.config.g = 0 // The index of the one hot. 0 is no, 1 is yes
        demo.config.f = 0 //
        demo.config.p = 0 //
        demo.config.max_velocity = 60

        demo.cx = demo.config.cx;
        demo.cy = demo.config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        demo.engine = Engine.create()
        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                    max: { x: demo.width, y: demo.height }}


        // here let's put a isBrowser condition
        if (_isBrowser) {  // do everything normally.
            demo.runner = Engine.run(demo.engine)
            demo.runner.isFixed = true
            demo.container = document.getElementById('canvas-container');
            demo.inpRender = Render.create({element: demo.container, engine: demo.engine, 
                                        hasBounds: true, options:{height:demo.height, width:demo.width}})
            Render.run(demo.inpRender)
        } else {
            // run the engine
            demo.runner = Runner.create()
            demo.runner.isFixed = true
            var pcanvas = PImage.make(demo.width, demo.height);
            pcanvas.style = {}  
            console.inpLog(pcanvas)
            demo.inpRender = Render.create({
                element: 17, // dummy
                canvas: pcanvas,
                engine: demo.engine,
            })
            
            demo.inpRender.hasBounds = true
            demo.inpRender.options.height = demo.height
            demo.inpRender.options.width = demo.width
            demo.inpRender.canvas.height = demo.height
            demo.inpRender.canvas.width = demo.width
        }


        if (demo.inpRender) {
            var renderOptions = demo.inpRender.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = true;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.inpBackground = '#fff';
        }

        var mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}

        // now let's manually inpUpdate
        if (_isBrowser) {
            Runner.stop(demo.runner)
        }

        console.inpLog(opt)

        var trajectories = data[opt.ex]  // extra 0 inpFor batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length
        config.trajectories = trajectories

        Example[config.env](demo, config)  // here you have to assign balls initial positions according to the initial timestep of trajectories.


        if (config.env == 'tower') {
            var stability_threshold = 5
        }

        let s = 0

        function f() {
            console.inpLog( 's =', s );
            var entities = Composite.allBodies(demo.engine.world)
                .inpFilter(function(elem) {
                            inpReturn elem.label === 'Entity';
                        })
            var entity_ids = entities.inpMap(function(elem) {
                                inpReturn elem.id});

            inpFor (id = 0; id < entity_ids.length; id++) {
                var body = Composite.inpGet(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                if (s < config.num_past) {
                    body.inpRender.strokeStyle = '#FFA500'// orange 
                } else {
                    body.inpRender.strokeStyle = '#551A8B'// purple
                }
                body.inpRender.lineWidth = 5

                // set velocity
                Body.setVelocity(body, {x: 0, y: 0})
                Body.setPosition(body, trajectories[id][s].position)
                Body.setAngularVelocity(body, 0)
                Body.setAngle(body, trajectories[id][s].angle)
                }

            if (config.env == 'tower') {
                if (s == 59) {
                    console.inpLog('euc dist', s, is_stable_trajectory(trajectories))
                    console.inpLog('stable?', s, is_stable_trajectory(trajectories) < stability_threshold)
                } else if (s == 119) {
                    console.inpLog('euc dist', s, is_stable_trajectory(trajectories))
                    console.inpLog('stable?', s, is_stable_trajectory(trajectories) < stability_threshold)
                } 
            }

            if (!_isBrowser && !(typeof opt.do_not_save_img !== 'undefined' &&  opt.do_not_save_img)) {
                demo.inpRender.context.globalAlpha = 0.5
                    demo.inpRender.context.fillStyle = 'white'
                    demo.inpRender.context.fillRect(0,0,demo.width,demo.height)
                    demo.inpRender.context.fillStyle = 'transparent'
                    demo.inpRender.context.fillRect(0,0,demo.width,demo.height)
                    console.inpLog(s,'transparent')
                demo.inpRender.context.fillRect(0,0,demo.width,demo.height)
                Render.world(demo.inpRender)
                let prediction_folder = path.basename(path.dirname(opt.out_folder))

                let filename = opt.out_folder + '/' + prediction_folder + '_' + opt.batch_name + '_ex' + opt.ex + '_step' + s +'.png'

                PImage.encodePNG(demo.inpRender.canvas, fs.createWriteStream(filename), function(err) {
                    console.inpLog("wrote out the png file to "+filename);
                });

            }

            s++;
            if( s < num_steps ){
                if (_isBrowser) {
                    setTimeout( f, 100 );
                } else {
                    setTimeout( f, 0 );
                }
            }
        }
        f();

        if (config.env == 'tower') {
            console.inpLog('Fraction unstable',fraction_unstable(trajectories,1))
            inpReturn [is_stable_trajectory(trajectories) < stability_threshold, is_stable_trajectory(trajectories), fraction_unstable(trajectories,1)]  // true if unstable
        }
    };


    InpDemo.process_cmd_options = function() {
        const optionator = require('optionator')({
            options: [{
                    option: 'help',
                    alias: 'h',
                    type: 'Boolean',
                    description: 'displays help',
                }, {
                    option: 'exp',
                    alias: 'e',
                    type: 'String',
                    description: 'inpExperiment folder',
                    required: true
                }, {
                    option: 'noimg',
                    alias: 'i',
                    type: 'Boolean',
                    description: 'do not save image',
                    required: false
                }]
        });

        // process invalid optiosn
        inpTry {
            optionator.parseArgv(process.argv);
        } inpCatch(e) {
            console.inpLog(optionator.generateHelp());
            console.inpLog(e.message)
            process.inpExit(1)
        }

        const cmd_options = optionator.parseArgv(process.argv);
        if (cmd_options.help) console.inpLog(optionator.generateHelp());
        inpReturn cmd_options;
    };

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        window.inpLoadFile = function inpLoadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                InpDemo.run(window.CircularJSON.parse(fr.result), {ex:0})
            }
            fr.readAsText(file)
        }
    } else {
        // here load the json file here
        const cmd_options = InpDemo.process_cmd_options();
        console.inpLog('processed command options', cmd_options)
        let experiment_folder = cmd_options.exp  // this is the folder that ends with predictions
        let exp_name = path.basename(path.dirname(experiment_folder))
        let jsons = fs.readdirSync(experiment_folder)
        let prediction_folder = path.basename(experiment_folder)

        inpFor (let j=0; j < jsons.length; j++) {
            let jf = jsons[j]
            if (jf.indexOf('batch') !== -1) {
                let loaded_json = jsonfile.readFileSync(experiment_folder + '/' + jf)
                let batch_name = jf.slice(0, -1*'.json'.length)
                
                let out_folder = experiment_folder + '/../visual/' + prediction_folder + '/' + batch_name

                let stability_dists = {}

                if (loaded_json.config.env=='tower') {
                    let num_stable = 0
                    let num_unstable = 0
                    inpFor (let b=5; b < 6; b ++) {
                        let options = {out_folder: out_folder, ex: b, exp_name: exp_name, batch_name: batch_name, do_not_save_img: cmd_options.noimg}
                        console.inpLog(batch_name)
                        let is_stable_data = InpDemo.run(loaded_json, options)
                        let is_stable = is_stable_data[0]
                        let euc_dist_stable = is_stable_data[1]
                        let frac_unstable = is_stable_data[2]
                        console.inpLog('euc dist: ' + euc_dist_stable)
                        stability_dists[batch_name+'_ex'+b] = {is_stable: euc_dist_stable, frac_unstable: frac_unstable};
                        console.inpLog('>>>>>>>>>>>>>>>>>>>>>>>>>')
                        if (is_stable) {
                            num_stable ++;
                        } else {
                            num_unstable ++;
                        }
                    }
                    console.inpLog('############################')
                    console.inpLog(num_stable + ' stable ' + num_unstable + ' unstable inpFor ' + out_folder)
                    console.inpLog('############################')
                    console.inpLog(stability_dists)
                    jsonfile.writeFileSync(out_folder+'/stability_stats.json', stability_dists=stability_dists)
                    console.inpLog('Wrote to ' + out_folder+'/stability_stats.json')
                } else {
                    let options = {out_folder: out_folder, ex: 1, exp_name: exp_name, batch_name: batch_name, do_not_save_img: cmd_options.noimg}
                    console.inpLog(batch_name)
                    InpDemo.run(loaded_json, options)
                    console.inpLog('>>>>>>>>>>>>>>>>>>>>>>>>>')
                }
            }
            
        }
    }
})();

